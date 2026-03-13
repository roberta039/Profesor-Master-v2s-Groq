import streamlit as st
import json
import streamlit.components.v1 as components
from groq import Groq as GroqClient
from supabase import create_client, Client
import uuid
import time
import tempfile
import os
import random
import re
import hashlib




# === APP INSTANCE ID ===
# Separă datele între instanțe diferite ale aceleiași aplicații (același Supabase, app-uri diferite)
# Setează APP_INSTANCE_ID în secrets.toml: APP_INSTANCE_ID = "profesor_v1"
_APP_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')

@st.cache_data(ttl=3600)
def get_app_id() -> str:
    """Returnează ID-ul aplicației. Validat anti-injection.
    FIX 6: cache-uit cu st.cache_data — st.secrets accesează discul la fiecare apel,
    iar get_app_id() e apelat la fiecare query Supabase.
    """
    try:
        raw = str(st.secrets.get("APP_INSTANCE_ID", "default")).strip() or "default"
    except Exception:
        raw = "default"
    return raw if _APP_ID_PATTERN.match(raw) else "default"

# === CONSTANTE PENTRU LIMITE (FIX MEMORY LEAK) ===
MAX_MESSAGES_IN_MEMORY = 100
MAX_MESSAGES_TO_SEND_TO_AI = 20
MAX_MESSAGES_IN_DB_PER_SESSION = 500
CLEANUP_DAYS_OLD = 90  # Păstrăm istoricul 90 de zile — acoperă vacanțe, pauze lungi

# === MODEL GROQ — singura sursă de adevăr pentru numele modelului ===
GEMINI_MODEL = "llama-3.3-70b-versatile"  # Groq: model principal
SUMMARIZE_AFTER_MESSAGES = 30   # Rezumăm când depășim acest număr de mesaje
MESSAGES_KEPT_AFTER_SUMMARY = 10  # Câte mesaje recente păstrăm după rezumare

# === ISTORIC CONVERSAȚII ===
def get_session_list(limit: int = 20) -> list[dict]:
    """Returnează lista sesiunilor folosind view-ul session_previews din Supabase.

    Un singur query în loc de două — agregarea se face în DB, nu în Python.
    View-ul returnează direct: session_id, app_id, last_active, msg_count, preview.

    Cache: invalidat imediat după orice modificare (mesaj nou, sesiune ștearsă etc.)
    """
    cache_ts  = st.session_state.get("_sess_list_ts", 0)
    cache_val = st.session_state.get("_sess_list_cache", None)
    force_refresh = st.session_state.get("_sess_cache_dirty", False)
    if force_refresh:
        st.session_state["_sess_cache_dirty"] = False

    if not force_refresh and cache_val is not None and (time.time() - cache_ts) < 5:
        return cache_val

    try:
        supabase = get_supabase_client()

        # Un singur query pe view-ul session_previews (agregare în DB)
        resp = (
            supabase.table("session_previews")
            .select("session_id, last_active, msg_count, preview")
            .eq("app_id", get_app_id())
            .gt("msg_count", 0)
            .order("last_active", desc=True)
            .limit(limit)
            .execute()
        )
        result = resp.data or []

        st.session_state["_sess_list_cache"] = result
        st.session_state["_sess_list_ts"]    = time.time()
        return result

    except Exception as e:
        _log("Eroare la încărcarea sesiunilor", "silent", e)
        return cache_val or []


def _cleanup_gfiles() -> None:
    """Curăță referințele la fișiere din session_state.
    (Groq nu folosește Google Files API — funcție păstrată pentru compatibilitate.)
    """
    gfile_keys = [k for k in st.session_state.keys() if k.startswith("_gfile_")]
    for k in gfile_keys:
        st.session_state.pop(k, None)


def switch_session(new_session_id: str):
    """Comută la o altă sesiune."""
    _cleanup_gfiles()  # curățăm fișierele Google la switch sesiune
    st.session_state.session_id = new_session_id
    st.session_state.messages = []
    invalidate_session_cache()  # FIX: forțează refresh la switch
    # FIX: curățăm summary-ul și cheile de mismatch la switch sesiune
    # altfel contextul sesiunii vechi e injectat în cea nouă
    st.session_state.pop("_conversation_summary", None)
    st.session_state.pop("_summary_cached_at", None)
    st.session_state.pop("_summary_for_sid", None)
    # Curățăm toate cheile _mismatch_warned_* (una per sesiune anterioară)
    for _k in [k for k in st.session_state.keys() if k.startswith("_mismatch_warned_")]:
        del st.session_state[_k]
    # Actualizează localStorage cu noul SID — JS-ul îl va folosi la următorul load
    components.html(
        f"<script>localStorage.setItem('profesor_session_id', {json.dumps(new_session_id)});</script>",
        height=0
    )


def invalidate_session_cache():
    """Marchează cache-ul sesiunilor ca expirat — apelat după orice modificare."""
    st.session_state["_sess_cache_dirty"] = True
    st.session_state["_sess_list_ts"] = 0  # FIX: resetează timestamp pentru forțare refresh complet


def format_time_ago(timestamp) -> str:
    """Formatează timestamp ca timp relativ (ex: '2 ore în urmă'). Acceptă float sau ISO string."""
    # FIX: Supabase poate returna ISO string în loc de float
    if isinstance(timestamp, str):
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            timestamp = dt.timestamp()
        except Exception:
            return "necunoscut"
    try:
        diff = time.time() - float(timestamp)
    except (TypeError, ValueError):
        return "necunoscut"
    if diff < 60:
        return "acum"
    elif diff < 3600:
        mins = int(diff / 60)
        return f"{mins} min în urmă"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours}h în urmă"
    else:
        days = int(diff / 86400)
        return f"{days} zile în urmă"




# === SUPABASE CLIENT + FALLBACK ===
@st.cache_resource(ttl=3600)  # Reîmprospătează la fiecare oră — previne token expiry
def get_supabase_client() -> Client | None:
    """Returnează clientul Supabase (conexiunea e lazy, fără query de test)."""
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def is_supabase_available() -> bool:
    """Returnează statusul Supabase din cache — nu face request la fiecare apel.
    Statusul se actualizează doar când o operație reală eșuează sau reușește."""
    return st.session_state.get("_sb_online", True)


def _mark_supabase_offline():
    """Marchează Supabase ca offline și notifică utilizatorul."""
    was_online = st.session_state.get("_sb_online", True)
    st.session_state["_sb_online"] = False
    if was_online:
        st.toast("⚠️ Baza de date offline — modul local activat.", icon="📴")


def _mark_supabase_online():
    """Marchează Supabase ca online și golește coada offline."""
    was_offline = not st.session_state.get("_sb_online", True)
    st.session_state["_sb_online"] = True
    if was_offline:
        st.toast("✅ Conexiunea restabilită!", icon="🟢")
        _flush_offline_queue()


# --- Coadă offline: mesaje salvate local când Supabase e down ---
MAX_OFFLINE_QUEUE_SIZE = 50  # Previne memory leak când Supabase e offline mult timp

def _get_offline_queue() -> list:
    queue = st.session_state.setdefault("_offline_queue", [])
    # Dacă coada depășește limita, păstrăm doar cele mai recente mesaje
    if len(queue) > MAX_OFFLINE_QUEUE_SIZE:
        st.session_state["_offline_queue"] = queue[-MAX_OFFLINE_QUEUE_SIZE:]
    return st.session_state["_offline_queue"]


def _flush_offline_queue():
    """Trimite mesajele din coada offline la Supabase când revine online.
    Anti-loop: dacă un mesaj eșuează de MAX_FLUSH_RETRIES ori, e abandonat.
    Anti-race: flag _flushing_queue previne procesarea dublă."""
    MAX_FLUSH_RETRIES = 3
    if st.session_state.get("_flushing_queue", False):
        return
    st.session_state["_flushing_queue"] = True
    # FIX 2: failed și queue inițializate înainte de try — garantat definite în finally/după
    failed = []
    queue = []
    try:
        queue = _get_offline_queue()
        if not queue:
            return
        client = get_supabase_client()
        if not client:
            return
        failed = []
        retry_counts = st.session_state.setdefault("_offline_retry_counts", {})
        for item in queue:
            item_key = f"{item.get('session_id','')}-{item.get('timestamp','')}"
            retries = retry_counts.get(item_key, 0)
            if retries >= MAX_FLUSH_RETRIES:
                _log(f"Mesaj abandonat după {MAX_FLUSH_RETRIES} încercări eșuate", "silent")
                continue
            try:
                client.table("history").insert(item).execute()
                retry_counts.pop(item_key, None)
            except Exception:
                retry_counts[item_key] = retries + 1
                failed.append(item)
        st.session_state["_offline_queue"] = failed
        st.session_state["_offline_retry_counts"] = retry_counts
    finally:
        st.session_state["_flushing_queue"] = False
    successful = len(queue) - len(failed)
    if successful > 0:
        st.toast(f"✅ {successful} mesaje sincronizate cu baza de date.", icon="☁️")

st.set_page_config(page_title="Profesor Liceu", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# Aplică tema dark/light imediat la fiecare rerun
if st.session_state.get("dark_mode", False):
    st.markdown("""
    <script>
    (function() {
        function applyDark() {
            const root = window.parent.document.documentElement;
            root.setAttribute('data-theme', 'dark');
            // Streamlit's internal theme toggle
            const btn = window.parent.document.querySelector('[data-testid="baseButton-headerNoPadding"]');
        }
        applyDark();
        // Re-apply after Streamlit re-renders
        setTimeout(applyDark, 100);
        setTimeout(applyDark, 500);
    })();
    </script>
    <style>
        /* Manual dark mode overrides pentru elementele principale */
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22 !important;
        }
        .stChatMessage {
            background-color: #1a1f2e !important;
        }
        .stTextArea textarea, .stTextInput input {
            background-color: #1a1f2e !important;
            color: #fafafa !important;
            border-color: #444 !important;
        }
        .stSelectbox > div, .stRadio > div {
            background-color: #1a1f2e !important;
            color: #fafafa !important;
        }
        p, h1, h2, h3, h4, h5, h6, li, label, span {
            color: #fafafa !important;
        }
        .stButton > button {
            border-color: #555 !important;
        }
        hr { border-color: #333 !important; }
        .stExpander { border-color: #333 !important; }
        [data-testid="stChatInput"] {
            background-color: #1a1f2e !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    .stChatMessage { font-size: 16px; }
    footer { visibility: hidden; }

    /* SVG container - light mode */
    .svg-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
        margin: 15px 0;
        overflow: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100%;
    }
    .svg-container svg { max-width: 100%; height: auto; }

    /* Dark mode */
    [data-theme="dark"] .svg-container {
        background-color: #1e1e2e;
        border-color: #444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }



    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 4px;
        font-size: 14px;
        color: #888;
    }
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    .typing-dots span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #888;
        animation: typing-bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
        40%            { transform: scale(1.0); opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)


# === DATABASE FUNCTIONS (SUPABASE) ===

# ÎMBUNĂTĂȚIRE 3: Logger centralizat — afișează toast utilizatorului ȘI loghează în consolă.
# Niveluri: "info" (toast albastru), "warning" (toast portocaliu), "error" (toast roșu).
# Erorile silențioase de fundal (cleanup, trim) folosesc doar consola.
def _log(msg: str, level: str = "silent", exc: Exception = None):
    """Loghează un mesaj și opțional afișează un toast în interfață.
    
    level:
        "silent"  — doar print în consolă (erori de fundal, nu deranjează utilizatorul)
        "info"    — toast verde, pentru operații reușite/informative
        "warning" — toast portocaliu, pentru degradări non-critice
        "error"   — toast roșu, pentru erori vizibile utilizatorului
    """
    full_msg = f"{msg}: {exc}" if exc else msg
    print(full_msg)
    icon_map = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
    if level in icon_map:
        try:
            st.toast(msg, icon=icon_map[level])
        except Exception:
            pass  # st.toast poate eșua în contexte fără sesiune activă


def init_db():
    """Verifică conexiunea la Supabase. Dacă e offline, activează modul local."""
    online = is_supabase_available()
    if not online:
        st.warning("📴 **Modul offline activ** — conversația se păstrează în memorie. "
                   "Istoricul va fi sincronizat automat când conexiunea revine.", icon="⚠️")


def cleanup_old_sessions(days_old: int = CLEANUP_DAYS_OLD):
    """Șterge sesiunile vechi — rulează cel mult o dată pe zi.
    Șterge HISTORY înainte de SESSIONS (ordinea corectă pentru integritate DB)."""
    if time.time() - st.session_state.get("_last_cleanup", 0) < 86400:
        return
    st.session_state["_last_cleanup"] = time.time()
    try:
        supabase = get_supabase_client()
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        supabase.table("history").delete().lt("timestamp", cutoff_time).eq("app_id", get_app_id()).execute()
        supabase.table("sessions").delete().lt("last_active", cutoff_time).eq("app_id", get_app_id()).execute()
    except Exception as e:
        _log("Eroare la curățarea sesiunilor vechi", "silent", e)


def save_message_to_db(session_id, role, content):
    """Salvează un mesaj în Supabase. Dacă e offline, pune în coada locală."""
    record = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "app_id": get_app_id()
    }
    if not is_supabase_available():
        q = _get_offline_queue()
        if len(q) < MAX_OFFLINE_QUEUE_SIZE:
            q.append(record)
        return
    try:
        client = get_supabase_client()
        client.table("history").insert(record).execute()
        _mark_supabase_online()
    except Exception as e:
        _log("Mesajul nu a putut fi salvat", "warning", e)
        _mark_supabase_offline()
        q = _get_offline_queue()
        if len(q) < MAX_OFFLINE_QUEUE_SIZE:
            q.append(record)


def load_history_from_db(session_id, limit: int = MAX_MESSAGES_IN_MEMORY):
    """Încarcă istoricul din Supabase. Fallback: returnează ce e deja în session_state.
    
    Când e offline: afișează avertisment și marchează că istoricul e incomplet
    (poate diferi de ce e în DB dacă utilizatorul a șters sau a schimbat sesiunea).
    """
    if not is_supabase_available():
        # FIX bug 12: offline → returnăm TOATE mesajele din memorie (nu trunchiate la limit)
        # limit-ul e pentru DB unde stocăm mult; în memorie avem deja mesajele relevante
        st.session_state["_history_may_be_incomplete"] = True
        return st.session_state.get("messages", [])
    try:
        client = get_supabase_client()
        response = (
            client.table("history")
            .select("role, content, timestamp")
            .eq("session_id", session_id)
            .eq("app_id", get_app_id())
            .order("timestamp", desc=False)
            .limit(limit)
            .execute()
        )
        return [{"role": row["role"], "content": row["content"]} for row in response.data]
    except Exception as e:
        _log("Eroare la încărcarea istoricului", "silent", e)
        return st.session_state.get("messages", [])[-limit:]


def clear_history_db(session_id):
    """Șterge istoricul pentru o sesiune din Supabase."""
    if not is_valid_session_id(session_id):
        _log(f"clear_history_db: session_id invalid ignorat: {str(session_id)[:20]}", "warning")
        return
    try:
        supabase = get_supabase_client()
        supabase.table("history").delete().eq("session_id", session_id).eq("app_id", get_app_id()).execute()
        invalidate_session_cache()  # FIX: sesiune ștearsă = cache invalid
        # Invalidăm și cache-ul rezumatului — conversația e nouă
        st.session_state.pop("_conversation_summary", None)
        st.session_state.pop("_summary_cached_at", None)
        st.session_state.pop("_summary_for_sid", None)
        st.session_state.pop("_mismatch_warned", None)
    except Exception as e:
        _log("Istoricul nu a putut fi șters", "warning", e)


def trim_db_messages(session_id: str):
    """Limitează mesajele din DB pentru o sesiune (FIX MEMORY LEAK)."""
    try:
        supabase = get_supabase_client()

        # Numără mesajele sesiunii
        count_resp = (
            supabase.table("history")
            .select("id", count="exact")
            .eq("session_id", session_id)
            .eq("app_id", get_app_id())
            .execute()
        )
        count = count_resp.count or 0

        if count > MAX_MESSAGES_IN_DB_PER_SESSION:
            to_delete = count - MAX_MESSAGES_IN_DB_PER_SESSION
            # Obține ID-urile celor mai vechi mesaje
            old_resp = (
                supabase.table("history")
                .select("id")
                .eq("session_id", session_id)
                .eq("app_id", get_app_id())
                .order("timestamp", desc=False)
                .limit(to_delete)
                .execute()
            )
            ids_to_delete = [row["id"] for row in old_resp.data]
            if ids_to_delete:
                supabase.table("history").delete().in_("id", ids_to_delete).execute()
    except Exception as e:
        _log("Eroare la curățarea DB", "silent", e)


# === SESSION MANAGEMENT (SUPABASE) ===
import secrets  # FIX bug 3: session IDs criptografic sigure

def generate_unique_session_id() -> str:
    """Generează un session ID criptografic sigur, fără risc de coliziuni.
    FIX bug 3: secrets.token_hex(32) = 64 caractere hex, entropie 256 biți —
    mult mai sigur decât combinația uuid[:16]+time+uuid[:8] anterioară."""
    return secrets.token_hex(32)  # 64 caractere hex lowercase, validat de _SESSION_ID_RE


# Regex precompilat pentru validarea session_id — doar hex lowercase, 16-64 caractere
_SESSION_ID_RE = re.compile(r'^[a-f0-9]{16,64}$')

def is_valid_session_id(sid: str) -> bool:
    """Validează session_id: doar hex lowercase, lungime 16-64 caractere.
    
    FIX: Fără validare, un sid malițios din URL (?sid=../../../etc) putea
    ajunge direct în query-urile Supabase ca parametru nevalidat.
    """
    if not sid or not isinstance(sid, str):
        return False
    return bool(_SESSION_ID_RE.match(sid))


def session_exists_in_db(session_id: str) -> bool:
    """Verifică dacă un session_id există deja în Supabase."""
    try:
        supabase = get_supabase_client()
        response = (
            supabase.table("sessions")
            .select("session_id")
            .eq("session_id", session_id)
            .eq("app_id", get_app_id())
            .limit(1)
            .execute()
        )
        return len(response.data) > 0
    except Exception:
        return False


def register_session(session_id: str):
    """Înregistrează o sesiune nouă în Supabase. Silent dacă offline."""
    if not is_supabase_available():
        return
    try:
        client = get_supabase_client()
        now = time.time()
        client.table("sessions").upsert({
            "session_id": session_id,
            "created_at": now,
            "last_active": now,
            "app_id": get_app_id()
        }).execute()
    except Exception as e:
        _log("Eroare la înregistrarea sesiunii", "silent", e)


def update_session_activity(session_id: str):
    """Actualizează timestamp-ul activității — cel mult o dată la 5 minute."""
    last = st.session_state.get("_last_activity_update", 0)
    if time.time() - last < 300:
        return
    st.session_state["_last_activity_update"] = time.time()
    if not is_supabase_available():
        return
    try:
        client = get_supabase_client()
        client.table("sessions").update({
            "last_active": time.time()
        }).eq("session_id", session_id).execute()
    except Exception as e:
        _log("Eroare la actualizarea sesiunii", "silent", e)


def inject_session_js():
    """
    JS care sincronizează SID-ul între Python și localStorage al browserului.

    Flux prima vizită:
      Python generează SID nou → pune în ?sid= → JS îl citește din URL → salvează în localStorage
      → Python șterge ?sid= din URL la următorul rerun (SID e deja în session_state)

    Flux revenire (restart, tab nou):
      localStorage are SID → JS pune ?sid= în URL → Python îl citește la startup →
      restaurează sesiunea existentă din Supabase

    JS NU generează niciodată SID-uri — doar le persistă.
    """
    current_sid = st.session_state.get("session_id", "")
    components.html(f"""
    <script>
    (function() {{
        const SID_KEY    = 'profesor_session_id';
        const APIKEY_KEY = 'profesor_api_key';
        const params     = new URLSearchParams(window.parent.location.search);
        const sidInUrl   = params.get('sid');
        const pythonSid  = {json.dumps(current_sid)};

        // ── Dacă Python a pus un SID nou în URL, salvează-l în localStorage ──
        if (sidInUrl && sidInUrl.length >= 16) {{
            localStorage.setItem(SID_KEY, sidInUrl);
            // Curăță URL-ul vizual (SID-ul e deja salvat)
            params.delete('sid');
            params.delete('apikey');
            const newUrl = window.parent.location.pathname +
                (params.toString() ? '?' + params.toString() : '');
            window.parent.history.replaceState(null, '', newUrl);
        }}

        // ── Dacă Python nu are SID în URL dar localStorage are unul, trimite-l via URL ──
        // Asta acoperă cazul revenirii după restart/tab nou când Python nu știe SID-ul
        if (!sidInUrl && (!pythonSid || pythonSid.length < 16)) {{
            const storedSid = localStorage.getItem(SID_KEY);
            if (storedSid && storedSid.length >= 16) {{
                params.set('sid', storedSid);
                params.delete('apikey');
                const newUrl = window.parent.location.pathname + '?' + params.toString();
                // replaceState + forțăm rerun prin schimbarea URL-ului
                window.parent.history.replaceState(null, '', newUrl);
                // Forțăm Streamlit să citească noul ?sid= — schimbăm URL și facem reload
                window.parent.location.href = newUrl;
            }}
        }}

        // ── API key via postMessage ──
        const storedKey = localStorage.getItem(APIKEY_KEY);
        if (storedKey && storedKey.startsWith('AIza')) {{
            window.parent.postMessage({{ type: 'profesor_apikey', key: storedKey }}, '*');
        }}
    }})();
    </script>

    <script>
    window._saveApiKeyToStorage = function(key) {{
        if (key && key.startsWith('AIza')) {{
            localStorage.setItem('profesor_api_key', key);
        }}
    }};
    window._clearStoredApiKey = function() {{
        localStorage.removeItem('profesor_api_key');
    }};
    </script>
    """, height=0)


def get_or_create_session_id() -> str:
    """
    URL-ul ?sid= este SINGURA sursă de adevăr pentru identitatea browserului.

    PROBLEMA REZOLVATĂ: st.session_state poate fi shared între vizitatori pe aceeași
    instanță Streamlit. De aceea NU folosim session_state ca sursă primară — doar URL-ul.

    Flux prima vizită (URL fără ?sid=):
      Python generează UUID → îl pune în ?sid= → URL-ul devine unic per browser

    Flux revenire (bookmark, restart telefon):
      Elevul deschide URL-ul cu ?sid= → Python îl citește → restaurează istoricul
    """
    # Citește ?sid= din URL — sursa de adevăr
    sid_from_url = st.query_params.get("sid", "")

    if is_valid_session_id(sid_from_url):
        # URL are ?sid= valid — înregistrează dacă e nou, altfel restaurează
        if not session_exists_in_db(sid_from_url):
            register_session(sid_from_url)
        st.session_state["session_id"] = sid_from_url
        return sid_from_url

    # Nu există ?sid= valid în URL — prima vizită cu URL curat
    # NU citim din session_state — poate conține SID-ul altui utilizator
    new_id = generate_unique_session_id()
    register_session(new_id)
    try:
        st.query_params["sid"] = new_id
    except Exception:
        pass
    st.session_state["session_id"] = new_id
    return new_id


# === MEMORY MANAGEMENT (FIX MEMORY LEAK) ===
def trim_session_messages():
    """Limitează mesajele din session_state pentru a preveni memory leak.
    Păstrează primul mesaj (contextul inițial) — consistent cu get_context_for_ai."""
    if "messages" in st.session_state:
        current_count = len(st.session_state.messages)

        if current_count > MAX_MESSAGES_IN_MEMORY:
            excess = current_count - MAX_MESSAGES_IN_MEMORY
            first_msg = st.session_state.messages[0] if st.session_state.messages else None
            st.session_state.messages = st.session_state.messages[excess:]
            # Re-inserează primul mesaj dacă nu e deja prezent (context inițial)
            if first_msg and (not st.session_state.messages or st.session_state.messages[0] != first_msg):
                st.session_state.messages.insert(0, first_msg)
            st.toast(f"📝 Am arhivat {excess} mesaje vechi pentru performanță.", icon="📦")


def summarize_conversation(messages: list) -> str | None:
    """Cere AI-ului să rezume conversația de până acum.
    
    Returnează textul rezumatului sau None dacă eșuează.
    Folosit pentru a comprima istoricul lung fără a pierde contextul.
    """
    if not messages or len(messages) < 6:
        return None
    try:
        # Trimitem doar primele mesaje (cele care vor fi comprimate)
        msgs_to_summarize = messages[:-MESSAGES_KEPT_AFTER_SUMMARY]
        if len(msgs_to_summarize) < 4:
            return None

        history_for_summary = []
        for msg in msgs_to_summarize:
            role = "model" if msg["role"] == "assistant" else "user"
            history_for_summary.append({"role": role, "parts": [msg["content"][:500]]})

        summary_prompt = (
            "Fă un rezumat SCURT (maxim 200 cuvinte) al conversației de mai sus. "
            "Include: subiectele discutate, conceptele explicate, exercițiile rezolvate "
            "și orice context important despre nivelul și înțelegerea elevului. "
            "Scrie la persoana a 3-a: 'Elevul a întrebat despre... Am explicat...'"
        )
        chunks = list(run_chat_with_rotation(history_for_summary, [summary_prompt]))
        summary = "".join(chunks).strip()
        return summary if len(summary) > 20 else None
    except Exception:
        return None  # Eșec silențios — nu întrerupem conversația


def get_context_for_ai(messages: list) -> list:
    """Pregătește contextul pentru AI cu limită de mesaje.

    Strategie:
    1. Dacă există un rezumat pre-generat (din sesiune anterioară sau conversație lungă):
       → rezumat + ultimele MESSAGES_KEPT_AFTER_SUMMARY mesaje recente
       Aceasta acoperă și cazul "revenirii din altă zi" cu oricâte mesaje în istoric.
    2. Sub MAX_MESSAGES_TO_SEND_TO_AI mesaje și fără rezumat: trimite totul
    3. Peste SUMMARIZE_AFTER_MESSAGES și fără rezumat: generează rezumat acum
    4. Fallback: primul mesaj + ultimele MAX_MESSAGES_TO_SEND_TO_AI
    """
    # ── Cazul 1: există deja un rezumat (pre-generat la revenire SAU generat anterior) ──
    # Îl folosim indiferent de numărul de mesaje — e mai bun decât trunchiere brută
    cached_summary = st.session_state.get("_conversation_summary")
    cached_at      = st.session_state.get("_summary_cached_at", 0)

    if cached_summary:
        # Regenerăm rezumatul la fiecare 10 mesaje noi față de ultima rezumare
        if (len(messages) - cached_at) >= 10:
            new_summary = summarize_conversation(messages)
            if new_summary:
                cached_summary = new_summary
                st.session_state["_conversation_summary"] = new_summary
                st.session_state["_summary_cached_at"]    = len(messages)

        summary_msg = {
            "role": "user",
            "content": (
                "[CONTEXT CONVERSAȚIE ANTERIOARĂ — citește înainte de a răspunde]\n"
                f"{cached_summary}\n"
                "[MESAJE RECENTE — continuare directă]"
            )
        }
        summary_ack = {
            "role": "assistant",
            "content": "Am înțeles contextul. Continuăm de unde am rămas."
        }
        recent = messages[-MESSAGES_KEPT_AFTER_SUMMARY:]
        return [summary_msg, summary_ack] + recent

    # ── Cazul 2: conversație scurtă — trimitem totul ──
    if len(messages) <= MAX_MESSAGES_TO_SEND_TO_AI:
        return messages

    # ── Cazul 3: conversație lungă fără rezumat — generăm acum ──
    if len(messages) >= SUMMARIZE_AFTER_MESSAGES:
        summary = summarize_conversation(messages)
        if summary:
            st.session_state["_conversation_summary"] = summary
            st.session_state["_summary_cached_at"]    = len(messages)
            summary_msg = {
                "role": "user",
                "content": (
                    "[CONTEXT CONVERSAȚIE ANTERIOARĂ — citește înainte de a răspunde]\n"
                    f"{summary}\n"
                    "[MESAJE RECENTE — continuare directă]"
                )
            }
            summary_ack = {
                "role": "assistant",
                "content": "Am înțeles contextul. Continuăm de unde am rămas."
            }
            recent = messages[-MESSAGES_KEPT_AFTER_SUMMARY:]
            return [summary_msg, summary_ack] + recent

    # ── Cazul 4: fallback — primul mesaj + ultimele MAX_MESSAGES_TO_SEND_TO_AI ──
    first_message  = messages[0] if messages else None
    recent_messages = messages[-MAX_MESSAGES_TO_SEND_TO_AI:]
    if first_message and first_message not in recent_messages:
        return [first_message] + recent_messages
    return recent_messages


def save_message_with_limits(session_id: str, role: str, content: str):
    """Salvează mesaj și verifică limitele."""
    save_message_to_db(session_id, role, content)
    invalidate_session_cache()  # FIX: un mesaj nou înseamnă date noi în sidebar
    
    # Rulează trim în același thread — Streamlit nu e thread-safe
    # Rulăm la fiecare 50 mesaje pentru a nu bloca UI-ul la fiecare salvare
    if len(st.session_state.get("messages", [])) % 50 == 0:
        trim_db_messages(session_id)
    
    trim_session_messages()






# === SVG FUNCTIONS ===

# ÎMBUNĂTĂȚIRE 4: lxml pentru parsare și validare SVG robustă.
# Fallback automat la regex dacă lxml nu e disponibil.
try:
    from lxml import etree as _lxml_etree
    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False


def repair_unclosed_tags(svg_content: str) -> str:
    """Repară tag-uri SVG comune care nu sunt închise corect."""
    self_closing_tags = ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'image', 'use']
    
    for tag in self_closing_tags:
        # FIX: pattern mai robust — nu atinge tag-uri deja self-closing
        pattern = rf'<{tag}(\s[^>]*)?>(?!</{tag}>)'
        
        def fix_tag(match, _tag=tag):
            attrs = match.group(1) or ""
            # Dacă are deja / la final, e deja corect
            if attrs.rstrip().endswith('/'):
                return match.group(0)
            return f'<{_tag}{attrs}/>'
        
        svg_content = re.sub(pattern, fix_tag, svg_content)
    
    text_opens = len(re.findall(r'<text[^>]*>', svg_content))
    text_closes = len(re.findall(r'</text>', svg_content))
    
    if text_opens > text_closes:
        for _ in range(text_opens - text_closes):
            svg_content = svg_content.replace('</svg>', '</text></svg>')
    
    g_opens = len(re.findall(r'<g[^>]*>', svg_content))
    g_closes = len(re.findall(r'</g>', svg_content))
    
    if g_opens > g_closes:
        for _ in range(g_opens - g_closes):
            svg_content = svg_content.replace('</svg>', '</g></svg>')
    
    return svg_content



def repair_svg(svg_content: str) -> str:
    """Repară SVG incomplet sau malformat.

    ÎMBUNĂTĂȚIRE 4: Încearcă mai întâi repararea cu lxml (parser XML tolerant),
    care gestionează corect namespace-uri, encoding și structura arborescentă.
    Fallback la regex dacă lxml eșuează sau nu e disponibil.
    """
    if not svg_content:
        return None

    svg_content = svg_content.strip()

    # Pasul 1: asigură tag-uri <svg> deschis/închis
    has_svg_open  = bool(re.search(r'<svg[^>]*>', svg_content, re.IGNORECASE))
    has_svg_close = '</svg>' in svg_content.lower()

    if not has_svg_open:
        svg_content = (
            '<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" '
            'style="max-width:100%;height:auto;background-color:white;">\n'
            + svg_content + '\n</svg>'
        )
    elif has_svg_open and not has_svg_close:
        svg_content += '\n</svg>'

    if 'xmlns=' not in svg_content:
        svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if 'viewBox=' not in svg_content.lower():
        svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 800 600"', 1)

    # Pasul 2: repară cu lxml dacă e disponibil
    if _LXML_AVAILABLE:
        try:
            parser = _lxml_etree.XMLParser(
                recover=True,
                remove_comments=False,
                resolve_entities=False,
                ns_clean=True,
            )
            root = _lxml_etree.fromstring(svg_content.encode("utf-8"), parser)
            repaired = _lxml_etree.tostring(
                root,
                pretty_print=True,
                encoding="unicode",
                xml_declaration=False
            )
            return repaired
        except Exception:
            pass  # lxml a eșuat → continuăm cu fallback

    # Pasul 3: fallback regex
    svg_content = repair_unclosed_tags(svg_content)
    return svg_content


def validate_svg(svg_content: str) -> tuple:
    """Validează SVG și returnează (is_valid, error_message).

    ÎMBUNĂTĂȚIRE 4: Folosește lxml pentru validare structurală când e disponibil.
    """
    if not svg_content:
        return False, "SVG gol"

    visual_elements = ['path', 'rect', 'circle', 'ellipse', 'line', 'text', 'polygon', 'polyline', 'image']

    if _LXML_AVAILABLE:
        try:
            parser = _lxml_etree.XMLParser(recover=True)
            tree = _lxml_etree.fromstring(svg_content.encode("utf-8"), parser)
            has_content = any(f'<{el}' in svg_content.lower() for el in visual_elements)
            if not has_content:
                return False, "SVG fără elemente vizuale"
            return True, "OK"
        except Exception as xml_err:
            # lxml a eșuat complet — încercăm fallback simplu
            pass

    # Fallback validare simplă
    if '<svg' not in svg_content.lower():
        return False, "Lipsește tag-ul <svg>"
    if '</svg>' not in svg_content.lower():
        return False, "Lipsește tag-ul </svg>"
    has_content = any(f'<{elem}' in svg_content.lower() for elem in visual_elements)
    if not has_content:
        return False, "SVG fără elemente vizuale"
    return True, "OK"


def sanitize_svg(svg_content: str) -> str:
    """Sanitizeaza SVG - elimina scripturi si event handlers (XSS prevention).
    
    Acopera: <script>, on* handlers (ghilimele/backtick), href=javascript:,
    use href=data:, style behavior/expression, <foreignObject>.
    """
    if not svg_content:
        return svg_content
    # Elimina <script> complet
    svg_content = re.sub(r'<script\b[^>]*>.*?</script\s*>', '', svg_content,
                         flags=re.DOTALL | re.IGNORECASE)
    # Elimina event handlers on* cu ghilimele duble
    svg_content = re.sub(r'\s+on[a-zA-Z]+\s*=\s*"[^"]*"', '', svg_content)
    # Elimina event handlers on* cu ghilimele simple
    svg_content = re.sub(r"\s+on[a-zA-Z]+\s*=\s*'[^']*'", '', svg_content)
    # Elimina event handlers on* cu backtick (template literals)
    svg_content = re.sub(r'\s+on[a-zA-Z]+\s*=\s*`[^`]*`', '', svg_content)
    # Elimina href=javascript: si xlink:href=javascript:
    svg_content = re.sub(r'(xlink:)?href\s*=\s*["\']?\s*javascript:[^"\'>\s]*["\']?', '',
                         svg_content, flags=re.IGNORECASE)
    # Elimina <use href="data:..."> — poate injecta SVG/HTML extern
    svg_content = re.sub(r'<use\b[^>]*href\s*=\s*["\']data:[^"\']*["\'][^>]*>', '',
                         svg_content, flags=re.IGNORECASE)
    # Elimina style cu behavior: sau expression( (vector de atac IE/vechi)
    svg_content = re.sub(r'style\s*=\s*["\'][^"\']*(?:behavior|expression)\s*:[^"\']*["\']', '',
                         svg_content, flags=re.IGNORECASE)
    # Elimina <foreignObject> — permite injectare HTML arbitrar in SVG
    svg_content = re.sub(r'<foreignObject\b.*?</foreignObject\s*>', '', svg_content,
                         flags=re.DOTALL | re.IGNORECASE)
    return svg_content



def _is_gfile_active(gfile) -> bool:
    """Verifică dacă un fișier Google este activ — helper consistent folosit peste tot."""
    state_str = str(gfile.state)
    state_name = getattr(gfile.state, "name", "")
    return state_str in ("FileState.ACTIVE", "ACTIVE") or state_name == "ACTIVE"


def render_message_with_svg(content: str):
    """Renderează mesajul cu suport îmbunătățit pentru SVG."""
    has_svg_markers = '[[DESEN_SVG]]' in content
    # Regex precis: detectează doar blocuri SVG complete, nu menționări în text
    # FIX bug 3: \b word boundary corect — previne match pe tag-uri ca <svgfoo>
    has_svg_elements = bool(re.search(r'<svg\b[^>]*>.*?</svg\s*>', content, re.DOTALL | re.IGNORECASE))
    has_svg_sub_elements = any(tag in content.lower() for tag in ['<path', '<rect', '<circle', '<line', '<polygon'])
    
    if has_svg_markers or (has_svg_elements) or (has_svg_sub_elements and 'stroke=' in content):
        svg_code = None
        before_text = ""
        after_text = ""
        
        if '[[DESEN_SVG]]' in content:
            parts = content.split('[[DESEN_SVG]]')
            before_text = parts[0]
            if len(parts) > 1 and '[[/DESEN_SVG]]' in parts[1]:
                inner_parts = parts[1].split('[[/DESEN_SVG]]')
                svg_code = inner_parts[0]
                after_text = inner_parts[1] if len(inner_parts) > 1 else ""
            elif len(parts) > 1:
                svg_code = parts[1]
        elif '<svg' in content.lower():
            svg_match = re.search(r'<svg.*?</svg>', content, re.DOTALL | re.IGNORECASE)
            if svg_match:
                svg_code = svg_match.group(0)
                before_text = content[:svg_match.start()]
                after_text = content[svg_match.end():]
            else:
                svg_start = content.lower().find('<svg')
                if svg_start != -1:
                    before_text = content[:svg_start]
                    svg_code = content[svg_start:]
        
        if svg_code:
            svg_code = sanitize_svg(svg_code)
            svg_code = repair_svg(svg_code)
            is_valid, error = validate_svg(svg_code)
            
            if is_valid:
                if before_text.strip():
                    st.markdown(before_text.strip())
                
                st.markdown(
                    f'<div class="svg-container">{svg_code}</div>',
                    unsafe_allow_html=True
                )
                
                if after_text.strip():
                    st.markdown(after_text.strip())
                return
            else:
                st.warning(f"⚠️ Desenul nu a putut fi afișat corect: {error}")
    
    clean_content = content
    clean_content = re.sub(r'\[\[DESEN_SVG\]\]', '\n🎨 *Desen:*\n', clean_content)
    clean_content = re.sub(r'\[\[/DESEN_SVG\]\]', '\n', clean_content)
    
    st.markdown(clean_content)


# === INIȚIALIZARE ===
init_db()
cleanup_old_sessions(CLEANUP_DAYS_OLD)

# Python generează/restaurează SID — poate pune ?sid= în URL pentru JS
session_id = get_or_create_session_id()
st.session_state.session_id = session_id
update_session_activity(session_id)

# JS citește ?sid= din URL (dacă Python l-a pus) și îl salvează în localStorage
# La revenire după restart: JS citește SID din localStorage și face reload cu ?sid=
inject_session_js()


# === API KEYS ===
#
# Prioritate:
#   1. Cheile din st.secrets (ale tale) — folosite primele, rotite automat
#   2. Cheia manuală a elevului din localStorage — folosită când ale tale
#      sunt epuizate SAU dacă nu ai setat nicio cheie în secrets
#
# Cheia elevului e salvată în localStorage al browserului său:
#   - supraviețuiește refresh-ului și închiderii tab-ului
#   - dispare doar dacă elevul apasă "Șterge cheia" sau golește browserul

# ── Pasul 1: citește cheia elevului din session_state (salvată direct, fără URL)
# FIX 1: cheia NU mai vine prin ?apikey= în URL — e salvată direct în session_state
# la click pe "Salvează cheia" și persistată în localStorage de JS via _saveApiKeyToStorage()
saved_manual_key = st.session_state.get("_manual_api_key", "")

# ── Pasul 2: construiește lista de chei (secrets + manuală) ──
raw_keys_secrets = None
if "GROQ_API_KEYS" in st.secrets:
    raw_keys_secrets = st.secrets["GROQ_API_KEYS"]
elif "GROQ_API_KEY" in st.secrets:
    raw_keys_secrets = [st.secrets["GROQ_API_KEY"]]
elif "GOOGLE_API_KEYS" in st.secrets:  # fallback dacă folosești același secrets.toml
    raw_keys_secrets = st.secrets["GOOGLE_API_KEYS"]
elif "GOOGLE_API_KEY" in st.secrets:
    raw_keys_secrets = [st.secrets["GOOGLE_API_KEY"]]

keys = []

# Adaugă cheile din secrets
if raw_keys_secrets:
    if isinstance(raw_keys_secrets, str):
        # Securitate: json.loads în loc de ast.literal_eval (mai sigur împotriva injection)
        import json as _json
        try:
            parsed = _json.loads(raw_keys_secrets)
            if isinstance(parsed, list):
                raw_keys_secrets = parsed
            else:
                raw_keys_secrets = [raw_keys_secrets]
        except (_json.JSONDecodeError, ValueError):
            # Fallback: split manual după virgulă, fără eval
            raw_keys_secrets = [k.strip().strip('"').strip("'")
                                 for k in raw_keys_secrets.split(",") if k.strip()]
    if isinstance(raw_keys_secrets, list):
        for k in raw_keys_secrets:
            if k and isinstance(k, str):
                clean_k = k.strip().strip('"').strip("'")
                if clean_k:
                    keys.append(clean_k)

# Adaugă cheia elevului la final (folosită când celelalte se epuizează)
if saved_manual_key and saved_manual_key not in keys:
    keys.append(saved_manual_key)

# ── Pasul 3: UI în sidebar pentru cheia manuală ──
# Afișăm secțiunea DOAR dacă nu există chei configurate în secrets
_are_secrets_keys = len([k for k in keys if k != saved_manual_key]) > 0

with st.sidebar:
    if not _are_secrets_keys:
        st.divider()
        st.subheader("🔑 Cheie API Groq")

        if not saved_manual_key:
            with st.expander("❓ Cum obțin o cheie? (gratuit)", expanded=False):
                st.markdown("**Ai nevoie de un cont Groq** (gratuit). Este complet gratuit.")
                st.markdown("**Pasul 1** — Deschide Groq Console:")
                st.link_button(
                    "🌐 Mergi la console.groq.com",
                    "https://console.groq.com/keys",
                    use_container_width=True
                )
                st.markdown("""
**Pasul 2** — Autentifică-te (cont Google sau email).

**Pasul 3** — Apasă **"Create API key"**.

**Pasul 4** — Copiază cheia afișată.
- Arată astfel: `gsk_...` (peste 50 caractere)

**Pasul 5** — Lipește cheia mai jos și apasă **Salvează**.

---
💡 **Limită gratuită:** 14.400 req/zi, 500.000 tokeni/minut — foarte generos.
                """)

            st.caption("Cheia se salvează în browserul tău și rămâne activă după refresh.")
            new_key = st.text_input(
                "Cheie API Groq:",
                type="password",
                placeholder="gsk_...",
                label_visibility="collapsed",
            )
            if st.button("✅ Salvează cheia", use_container_width=True, type="primary", key="save_api_key"):
                clean = new_key.strip().strip('"').strip("'")
                if clean and clean.startswith("gsk_") and len(clean) > 20:
                    st.session_state["_manual_api_key"] = clean
                    keys.append(clean)
                    components.html(
                        f"<script>window.parent._saveApiKeyToStorage && "
                        f"window.parent._saveApiKeyToStorage({json.dumps(clean)});</script>",
                        height=0
                    )
                    st.toast("✅ Cheie Groq salvată în browser!", icon="🔑")
                    st.rerun()
                else:
                    st.error("❌ Cheie invalidă. Trebuie să înceapă cu 'gsk_' și să aibă minim 20 caractere.")

        else:
            # Cheia e salvată — arată doar statusul și butonul de ștergere, fără ghid
            st.success("🔑 Cheie personală activă.")
            st.caption("Salvată în browserul tău — rămâne după refresh.")
            if st.button("🗑️ Șterge cheia", use_container_width=True, key="del_api_key"):
                st.session_state.pop("_manual_api_key", None)
                st.query_params.pop("apikey", None)
                # FIX 5: folosim components importat la nivel de modul
                components.html("<script>localStorage.removeItem('profesor_api_key');</script>", height=0)
                st.rerun()

if not keys:
    st.error("❌ Nicio cheie API Groq validă. Introdu cheia ta Groq (gsk_...) în bara laterală.")
    st.stop()

if "key_index" not in st.session_state:
    # Distribuie utilizatorii aleator între chei — nu toți pe cheia 0
    st.session_state.key_index = random.randint(0, max(len(keys) - 1, 0))
# Salvăm lista de chei în session_state — necesară pentru _cleanup_gfiles la switch sesiune
st.session_state["_api_keys_list"] = keys


# === MATERII ===
MATERII = {
    "🤖 Automat":         None,  # detectează materia din mesaj, întreabă dacă nu poate
    "📐 Matematică":      "matematică",
    "⚡ Fizică":          "fizică",
    "🧪 Chimie":          "chimie",
    "📖 Română":          "limba și literatura română",
    "🇫🇷 Franceză":       "limba franceză",
    "🇬🇧 Engleză":        "limba engleză",
    "🇩🇪 Germană":        "limba germană",
    "🌍 Geografie":       "geografie",
    "🏛️ Istorie":         "istorie",
    "💻 Informatică":     "informatică",
    "🧬 Biologie":        "biologie",
}

# Label-ul modului automat — folosit în mai multe locuri
_AUTOMAT_LABEL = "🤖 Automat"

# Mapare inversă cod → label (pentru toast-uri și afișări)
_MATERII_LABEL = {v: k for k, v in MATERII.items() if v is not None}



# ═══════════════════════════════════════════════════════════════
# PROMPT MODULAR — fiecare materie are blocul ei separat.
# get_system_prompt() include DOAR blocul materiei selectate,
# reducând tokenii de input cu 71-94% față de promptul complet.
# ═══════════════════════════════════════════════════════════════

_PROMPT_COMUN = r"""
    REGULI DE IDENTITATE (STRICT):
    1. Folosește EXCLUSIV genul masculin când vorbești despre tine.
       - Corect: "Sunt sigur", "Sunt pregătit", "Am fost atent", "Sunt bucuros".
       - GREȘIT: "Sunt sigură", "Sunt pregătită".
    2. Te prezinți simplu, fără nicio titulatură pompoasă.

    TON ȘI ADRESARE (CRITIC):
    3. Vorbește DIRECT, la persoana I singular.
       - CORECT: "Salut, sunt aici să te ajut." / "Te ascult." / "Sunt pregătit." / "Înțeleg!"
       - GREȘIT: "Înțeleg, Domnule Profesor!" / "Bineînțeles, Domnule Profesor!" / "Domnul profesor este aici." / "Profesorul te va ajuta."
       - NU folosi NICIODATĂ "Domnule Profesor" sau orice titulatură — tu ești profesorul, nu elevul.
    4. Fii cald, natural, apropiat și scurt. Evită introducerile pompoase.
    5. NU SALUTA în fiecare mesaj. Salută DOAR la începutul unei conversații noi.
    6. Dacă elevul pune o întrebare directă, răspunde DIRECT la subiect, fără introduceri de genul "Salut, desigur...".
    7. Folosește "Salut" sau "Te salut" în loc de formule foarte oficiale.

    REGULĂ STRICTĂ: Predă exact ca la școală (nivel Gimnaziu/Liceu).
    NU confunda elevul cu detalii despre "aproximări" sau "lumea reală" (frecare, erori) decât dacă problema o cere specific.


    ═══════════════════════════════════════════════
    STRATEGII DE ÎNVĂȚARE — COMPETENȚĂ OBLIGATORIE
    ═══════════════════════════════════════════════
    Ești expert nu doar în materii, ci și în CUM se învață eficient.
    Când elevul întreabă despre metode de studiu, organizare, concentrare sau blocaje,
    răspunzi ca un mentor experimentat — concret, personalizat, fără clișee.

    A. TEHNICI DE STUDIU:

       1. BLOCURI DE TIMP — 52+17 și 25+5 (Pomodoro)
          - 52 min lucru intens + 17 min pauză reală (fără telefon) = ciclu optim
          - 25+5 (Pomodoro clasic) = mai ușor când motivația e scăzută
          - În cele 52 min: un singur task, notificări OFF, telefon în altă cameră
          - Pauza: mișcare, apă, aer — NU social media (resetează creierul, nu îl obosește)
          - Dacă elevul e obosit → recomandă 25+5; dacă e în flux → 52+17

       2. ACTIVE RECALL (Recuperare activă) — cea mai eficientă tehnică
          - Citești o pagină → ÎNCHIZI cartea → reproduci din memorie
          - La exerciții: lucrezi tot ce știi FĂRĂ să te uiți la teorie, apoi revii la teorie
            exact pentru ce nu a ieșit — aceasta este Active Recall aplicat corect
          - De ce funcționează: creierul consolidează când *recuperează*, nu când *recitește*

       3. SPACED REPETITION (Repetiție eșalonată)
          - Curba Ebbinghaus: repeți la 1 zi → 3 zile → 7 zile → 21 zile = memorie permanentă
          - Practic: ce ai învățat luni revezi joi; ce ai văzut joi revezi săptămâna viitoare
          - Nu înghesuia tot într-o singură zi de studiu

       4. TEHNICA FEYNMAN
          - Studiezi conceptul → explici cu voce tare ca unui elev de cls. 5 → unde te blochezi
            = gaura în înțelegere → te întorci la sursă → simplifici până merge fără termeni tehnici
          - Nu poți explica ceea ce nu înțelegi cu adevărat

       5. INTERLEAVING (Intercalarea materiilor)
          - NU face 3 ore dintr-o materie continuu — alternează: fizică → matematică → fizică
          - Schimbarea contextului forțează creierul să reconstruiască conexiunile → mai solid
          - Excepție: când înveți ceva complet nou pentru prima dată → 1-2 ore blocat e ok

    B. STRUCTURA OPTIMĂ A UNUI BLOC DE 52 MINUTE:
       0-5 min:   Recapitulare rapidă — ce ai făcut în sesiunea anterioară (Active Recall)
       5-35 min:  Lucru intens — exerciții fără teorie (identifici ce știi și ce nu)
       35-45 min: Teoria exact pentru ce nu a ieșit — cauți specific, nu recitești tot
       45-50 min: Reîncerci exercițiile care nu au ieșit (cu teoria proaspătă)
       50-52 min: Notezi 3 lucruri cheie reținute (consolidare finală)

    C. ORGANIZAREA PE TERMEN LUNG:
       - Planifică săptămânal, nu zilnic (flexibilitate când apare ceva neprevăzut)
       - Max 2-3 materii/zi — focusul distribuit pe mai multe e mai puțin eficient
       - Identifică orele de vârf (dimineață sau seară?) → pune materiile grele acolo
       - Lasă 20% din timp neplanificat — buffer pentru ce durează mai mult

    D. BLOCAJ MENTAL ȘI ANXIETATE:
       - Blocat la o problemă > 10 minute → notezi unde te-ai oprit, treci mai departe
       - Anxietate înainte de examen: tehnica 4-7-8 (inspiră 4s, ține 7s, expiră 8s)
       - "Nu înțeleg nimic" = creier obosit, nu ești "prost" → pauză 20 min, problemă ușoară
       - Cu 2 zile înainte de BAC/teză: nu mai înveți lucruri noi, doar recapitulare ușoară

    E. SOMN, ALIMENTAȚIE, CONCENTRARE:
       - Somnul consolidează memoria — fără somn, studiul e pierdut parțial (minim 7-8 ore)
       - Hidratare: deshidratarea ușoară scade concentrarea cu ~20%
       - Nu studia imediat după masă grea — 20-30 min pauză
       - Mișcare fizică 20-30 min/zi crește BDNF → memorare mai bună

    F. APLICARE PRACTICĂ — RĂSPUNDE PERSONALIZAT:
       - Când elevul descrie rutina lui, ANALIZEZI ce face bine și ce poate îmbunătăți
       - Nu impui sistem rigid — adaptezi la contextul lui (ore, materii, nivel)
       - Când descrie că "lucrează ce știe, revine la teorie" — recunoști că e Active Recall și îi spui

    GHID DE COMPORTAMENT:"""

_PROMPT_FINAL = r"""
    11. STIL DE PREDARE:
           - Explică simplu, cald și prietenos. Evită "limbajul de lemn".
           - Folosește analogii pentru concepte grele (ex: "Curentul e ca debitul apei").
           - La teorie: Definiție → Exemplu Concret → Aplicație.
           - La probleme: Explică pașii logici ("Facem asta pentru că..."), nu da doar calculul.
           - Dacă elevul greșește: corectează blând, explică DE CE e greșit, dă exemplul corect.

    12. MATERIALE UPLOADATE (Cărți/PDF/Poze):
           - Dacă primești o poză sau un PDF, analizează TOT conținutul vizual înainte de a răspunde.
           - La poze cu probleme scrise de mână: transcrie problema, apoi rezolv-o.
           - Păstrează sensul original al textelor din manuale.

    13. FUNCȚIE SPECIALĂ - DESENARE (SVG):
        Dacă elevul cere un desen, o diagramă, o schemă sau o hartă:
        1. Ești OBLIGAT să generezi cod SVG valid.
        2. Codul trebuie încadrat STRICT între tag-uri:
           [[DESEN_SVG]]
           <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
              <!-- Codul tău aici -->
           </svg>
           [[/DESEN_SVG]]
        3. IMPORTANT: Nu uita tag-ul de deschidere <svg> și cel de închidere </svg>!
        4. Adaugă întotdeauna etichete text (<text>) pentru a numi elementele din desen.
        5. Folosește culori clare și contraste bune pentru lizibilitate.
        6. REGULĂ CRITICĂ: Dacă elevul cere DOAR desenul (ex: "fă-mi un desen", "arată-mi schema",
           "desenează"), răspunzi EXCLUSIV cu desenul SVG — fără explicații, fără rezolvare,
           fără text înainte sau după. DOAR blocul [[DESEN_SVG]]...[[/DESEN_SVG]].
           Dacă elevul vrea și explicații, le va cere separat.
"""

_PROMPT_SUBJECTS: dict[str, str] = {
    "matematică": r"""
    1. MATEMATICĂ — PROGRAMA OFICIALĂ 2026 (Liceu România):
       NOTAȚII OBLIGATORII (niciodată altele):
       - Derivată: f'(x) sau y' — NU dy/dx
       - Logaritm natural: ln(x) — NU log_e(x)
       - Logaritm zecimal: lg(x) — NU log(x), NU log_10(x)
       - Tangentă: tg(x) — NU tan(x)
       - Cotangentă: ctg(x) — NU cot(x)
       - Mulțimi: ℕ, ℤ, ℚ, ℝ, ℂ
       - Intervale: [a, b], (a, b), [a, b), (a, b]
       - Modul: |x| — NU abs(x)
       - Lucrează cu valori EXACTE (√2, π, e) — NICIODATĂ aproximații dacă nu se cere
       - Folosește LaTeX ($...$) pentru toate formulele

       📌 NOTĂ DE CLASĂ: La fiecare răspuns menționează clasa (IX/X/XI/XII) și dacă e
       trunchi comun (TC — toți elevii) sau curriculum specialitate (CS — profil real).

       ══════════════════════════════════════════
       CLASA A IX-A — Trunchi comun (toți elevii)
       ══════════════════════════════════════════

       LOGICĂ MATEMATICĂ:
       - Propoziții, predicate, valori de adevăr
       - Operații logice: negație (¬), conjuncție (∧), disjuncție (∨), implicație (⇒), echivalență (⟺)
       - Cuantificatori: ∀ (pentru orice), ∃ (există)
       - Reguli de negare: ¬(∀x P(x)) ↔ ∃x ¬P(x)
       - Demonstrații prin contradicție și contrapozitivă

       PROGRESII:
       - Progresie aritmetică: aₙ = a₁ + (n-1)r, Sₙ = n(a₁+aₙ)/2
       - Progresie geometrică: bₙ = b₁·qⁿ⁻¹, Sₙ = b₁(qⁿ-1)/(q-1)
       - Aplicații reale: rate, dobânzi simple și compuse
       - Recunoaștere tip din context: diferențe constante → aritmetică, rapoarte constante → geometrică

       GEOMETRIE ANALITICĂ ÎN PLAN:
       - Distanța: d(A,B) = √[(x₂-x₁)²+(y₂-y₁)²]
       - Mijlocul segmentului: M = ((x₁+x₂)/2, (y₁+y₂)/2)
       - Panta dreptei: m = (y₂-y₁)/(x₂-x₁)
       - Ecuația dreptei: y-y₁ = m(x-x₁) sau ax+by+c=0
       - Drepte paralele: m₁=m₂; drepte perpendiculare: m₁·m₂=-1
       - Ecuația cercului: (x-a)²+(y-b)²=r²

       FUNCȚII (introducere):
       - Domeniu de definiție: numitor≠0, radical≥0, logaritm>0
       - Monotonie: crescătoare/descrescătoare (din grafic sau derivată)
       - Paritate: f(-x)=f(x) → pară; f(-x)=-f(x) → impară
       - Tipuri: afine f(x)=ax+b, pătratice f(x)=ax²+bx+c, radical, exponențiale, logaritmice
       - Metodă grafic: domeniu → intersecții cu axe → monotonie → asimptote → grafic

       TRIGONOMETRIE:
       - Cercul trigonometric: raza 1, unghiuri în radiani și grade
       - Valori exacte OBLIGATORII:
         sin30°=1/2, cos30°=√3/2, tg30°=√3/3
         sin45°=√2/2, cos45°=√2/2, tg45°=1
         sin60°=√3/2, cos60°=1/2, tg60°=√3
         sin0°=0, cos0°=1, sin90°=1, cos90°=0
       - Identitate fundamentală: sin²x + cos²x = 1
       - Ecuații trigonometrice: formă canonică → soluție generală cu k∈ℤ

       ══════════════════════════════════════════
       CLASA A X-A — Trunchi comun (toți elevii)
       ══════════════════════════════════════════

       TRIGONOMETRIE APLICATĂ ÎN TRIUNGHIURI:
       - Teorema cosinusului: a² = b²+c²-2bc·cosA
       - Teorema sinusurilor: a/sinA = b/sinB = c/sinC = 2R
       - Rezolvarea triunghiurilor oarecare: identifică ce cunoști, alege formula potrivită
       - Aria triunghiului: S = (1/2)·b·c·sinA = (a·b·c)/(4R)

       COMBINATORICĂ (Metode de numărare):
       - Regula sumei și a produsului
       - Permutări: Pₙ = n!
       - Aranjamente: Aₙᵏ = n!/(n-k)!
       - Combinări: Cₙᵏ = n!/[k!(n-k)!]
       - Triunghiul lui Pascal: Cₙᵏ = Cₙ₋₁ᵏ⁻¹ + Cₙ₋₁ᵏ
       - Binomul lui Newton (n≤5): (a+b)ⁿ = Σ Cₙᵏ·aⁿ⁻ᵏ·bᵏ

       STATISTICĂ ȘI PROBABILITĂȚI:
       - Colectare și organizare date: tabele, frecvențe absolute/relative
       - Reprezentări grafice: diagrame bare, histograme, box-plot, diagrame circulare
       - Indicatori: medie aritmetică, mediană, mod, quartile Q1/Q2/Q3, abatere standard
       - Probabilitate: P(A) = cazuri favorabile / cazuri posibile
       - Evenimente disjuncte: P(A∪B) = P(A)+P(B)
       - Evenimente independente: P(A∩B) = P(A)·P(B)
       - Probabilitate condiționată: P(A|B) = P(A∩B)/P(B)

       FUNCȚII (continuare):
       - Studiu complet: funcție afină, pătratică, compuse
       - Interpretare grafice în context real: creșteri, descreșteri, maxime, minime
       - Operații cu funcții: sumă, produs, compunere

       ECUAȚII ȘI INECUAȚII (consolidare IX-X):
       - Ec. grad 1: ax+b=0 → x=-b/a
       - Ec. grad 2: Δ=b²-4ac, x₁,₂=(-b±√Δ)/2a
         → Δ<0: fără soluții reale; Δ=0: soluție dublă; Δ>0: două soluții
       - Inecuații grad 2: tabel de semne cu rădăcinile — NU formulă directă
       - Sisteme: substituție SAU reducere — arată explicit pașii

       ══════════════════════════════════════════
       CLASA A XI-A — Curriculum specialitate (profil real)
       ══════════════════════════════════════════

       MATRICE ȘI DETERMINANȚI:
       - Tipuri: matrice nulă, unitate, diagonală, simetrică, antisimetrică
       - Operații: adunare, scădere, înmulțire scalară, înmulțire matrice (AxB ≠ BxA!)
       - Determinant 2×2: det(A) = ad-bc
       - Determinant 3×3: dezvoltare după prima linie (regula Sarrus ca verificare)
       - Matrice inversabilă: det(A)≠0 → A⁻¹ = (1/det(A))·adj(A)
       - Aplicații: coliniaritate puncte, arie triunghi cu coordonate, rezolvare sisteme

       SISTEME LINIARE (XI):
       - Metoda lui Cramer: soluție unică când det(A)≠0
         → x = det(Aₓ)/det(A), y = det(Aᵧ)/det(A)
       - Regula: scrie matricea sistemului → calculează determinanți → soluție

       LIMITE ȘI CONTINUITATE:
       - Limita la un punct: încearcă substituție directă ÎNTÂI
       - Cazuri nedeterminate 0/0: factorizează sau folosește L'Hôpital
       - Cazuri ∞/∞: împarte la cea mai mare putere
       - Continuitate: f continuă în x₀ ↔ limₓ→ₓ₀f(x) = f(x₀)
       - Limite la ±∞: comportamentul asimptotic al funcției

       DERIVATE:
       - Definiție: f'(x₀) = lim[f(x₀+h)-f(x₀)]/h
       - Reguli de derivare (OBLIGATORII):
         (u±v)' = u'±v'
         (u·v)' = u'v + uv'
         (u/v)' = (u'v - uv')/v²
         (f∘g)'(x) = f'(g(x))·g'(x)  ← derivata funcției compuse
       - Derivate standard: (xⁿ)'=nxⁿ⁻¹, (eˣ)'=eˣ, (ln x)'=1/x,
         (sin x)'=cos x, (cos x)'=-sin x, (tg x)'=1/cos²x
       - APLICAȚII DERIVATE:
         → Monotonie: f'(x)>0 → crescătoare; f'(x)<0 → descrescătoare
         → Extreme locale: f'(x₀)=0 + schimbare semn → minim/maxim
         → Tabel de variație: obligatoriu pentru studiul complet al funcției
         → Optimizare: probleme practice (costuri minime, arii maxime, viteze)
         → Concavitate: f''(x)>0 → convexă; f''(x)<0 → concavă
         → Punct de inflexiune: f''(x₀)=0 și schimbare semn f''

       GEOMETRIE ÎN SPAȚIU (XI):
       - Reper cartezian Oxyz: coordonate puncte, vectori în spațiu
       - Distanța între două puncte în spațiu
       - Vectori: AB⃗ = (x₂-x₁, y₂-y₁, z₂-z₁)
       - Produs scalar: a⃗·b⃗ = axbx+ayby+azbz = |a⃗||b⃗|cosθ
       - Poziții relative: drepte și plane în spațiu
       - Distanța de la un punct la un plan
       - Volum tetraedru cu coordonate

       ══════════════════════════════════════════
       CLASA A XII-A — Curriculum specialitate (profil real)
       ══════════════════════════════════════════

       SISTEME LINIARE AVANSATE (XII):
       - Rangul unei matrice (metoda eliminării Gauss)
       - Clasificare sisteme: compatibil determinat (sol. unică), compatibil nedeterminat
         (infinit soluții), incompatibil (fără soluții) — pe baza rangurilor
       - Metoda Gauss (eliminare): matrice extinsă → formă treaptă → soluție
       - Teorema Kronecker-Capelli: rang(A)=rang(A|b) ↔ compatibil

       GEOMETRIE ÎN SPAȚIU (XII — continuare):
       - Ecuația planului: ax+by+cz+d=0
       - Plan determinat de 3 puncte (cu determinanți)
       - Distanța de la punct la plan: d = |ax₀+by₀+cz₀+d|/√(a²+b²+c²)
       - Unghiul dintre două plane, unghi dreaptă-plan
       - Calcule de volum: piramidă, con, sferă, cilindru

       PRIMITIVE ȘI INTEGRALE:
       - Primitivă: F'(x)=f(x) → F(x) = ∫f(x)dx + C
       - Primitive standard OBLIGATORII:
         ∫xⁿdx = xⁿ⁺¹/(n+1)+C (n≠-1)
         ∫(1/x)dx = ln|x|+C
         ∫eˣdx = eˣ+C
         ∫sin x dx = -cos x+C
         ∫cos x dx = sin x+C
         ∫(1/cos²x)dx = tg x+C
       - Metode de integrare:
         → Schimbare de variabilă: ∫f(g(x))·g'(x)dx — recunoaște tiparul
         → Integrare prin părți: ∫u·dv = uv - ∫v·du
       - INTEGRALA DEFINITĂ:
         → Formula Leibniz-Newton: ∫ₐᵇf(x)dx = F(b)-F(a)
         → Proprietăți: liniaritate, aditivitate, monotonie
       - APLICAȚII INTEGRALE:
         → Aria sub grafic: S = ∫ₐᵇ|f(x)|dx
         → Aria între două curbe: S = ∫ₐᵇ|f(x)-g(x)|dx
         → Volum de rotație în jurul axei Ox: V = π∫ₐᵇ[f(x)]²dx
         → Interpretare în fizică: lucru mecanic, cost total acumulat

       ══════════════════════════════════════════
       PROFILURI SPECIALE (când elevul menționează)
       ══════════════════════════════════════════

       PROFIL TEHNOLOGIC (programare liniară + grafuri):
       - Programare liniară: funcție obiectiv, restricții, poligon fezabil
         → Maximul/minimul se atinge într-un vârf al poligonului fezabil
       - Teoria grafurilor: noduri, muchii, grad, drum, ciclu
         → Matrice de adiacență, drum minim (Dijkstra)
         → Aplicații: rețele de transport, rețele de servicii

       PROFIL MATE-INFO (legătura matematică ↔ algoritmi):
       - Algoritmi numerici în Python: CMMDC (Euclid), Fibonacci, conversii baze
       - Implementare formule matematice: progresii, combinări, statistici
       - Vizualizare grafice cu matplotlib sau GeoGebra/Desmos
       - Verificare calcule matematice prin cod Python

       ══════════════════════════════════════════
       REGULI GENERALE MATEMATICĂ:
       ══════════════════════════════════════════
       - STRUCTURA obligatorie pentru probleme: Date → Formulă → Calcul → Răspuns
       - La funcții: ÎNTOTDEAUNA parcurge: domeniu → intersecții axe → monotonie → grafic
       - La geometrie: DESENEAZĂ (sau descrie) figura ÎNAINTE de calcul
       - La demonstrații: fiecare pas cu justificare din teoremă/definiție
       - LaTeX pentru toate formulele: $formula$ inline, $$formula$$ pe linie nouă
       - Valori EXACTE mereu: √2, π, e — NU 1.41, 3.14, 2.71
       - Unghiuri: dacă nu se specifică, lucrează în grade; menționează când folosești radiani
       - Verificare: la final verifică dacă răspunsul e plauzibil (semn, ordine mărime)""",
    "fizică": r"""
    2. FIZICĂ — PROGRAMA ROMÂNEASCĂ PE CLASE (CRITIC):

       NOTAȚII OBLIGATORII (toate clasele):
       - Viteză: v (nu V, nu velocity)
       - Accelerație: a (nu A)
       - Masă: m (nu M)
       - Forță: F (cu majusculă)
       - Timp: t (nu T — T e pentru perioadă)
       - Distanță/deplasare: d sau s sau x (conform problemei)
       - Energie cinetică: Ec = mv²/2 (NU ½mv²)
       - Energie potențială gravitațională: Ep = mgh
       - Lucru mecanic: L = F·d·cosα
       - Impuls: p = mv
       - Moment forță: M = F·d (brațul forței)

       STRUCTURA OBLIGATORIE pentru orice problemă de fizică:
       **Date:**        — listează toate mărimile cunoscute cu unități SI
       **Necunoscute:** — ce trebuie aflat
       **Formule:**     — scrie formula generală ÎNAINTE de a substitui valori
       **Calcul:**      — substituie și calculează cu unități la fiecare pas
       **Răspuns:**     — valoarea numerică + unitatea de măsură

       ══════════════════════════════════════════
       CLASA A IX-A — Mecanică + Mecanica fluidelor
       ══════════════════════════════════════════

       MĂSURĂRI ȘI ERORI:
       - Mărimi fizice, unități SI, instrumente de măsură
       - Eroare sistematică vs. aleatoare, incertitudine, notație științifică
       - Transformări de unități — obligatoriu pas explicit

       CINEMATICĂ:
       - Sistem de referință, traiectorie, vector poziție, deplasare vs. distanță
       - Viteză medie: v_m = Δx/Δt; viteză instantanee (tangenta la graficul x(t))
       - Accelerație medie: a_m = Δv/Δt
       - MRU: x = x₀ + v·t; grafic x(t) — dreaptă, grafic v(t) — orizontală
       - MRUV: v = v₀ + a·t; x = x₀ + v₀t + at²/2; v² = v₀² + 2aΔx
         → Alege formula care conține EXACT necunoscuta și datele cunoscute
         → NU deriva ecuațiile — folosește-le direct
       - Mișcare circulară uniformă: T, f, ω = 2π/T, v = ω·r, aₙ = v²/r = ω²·r

       DINAMICĂ NEWTONIANĂ:
       - Principiul I (inerției): corp fără forță netă → v = const
       - Principiul II: ΣF⃗ = m·a⃗ — suma VECTORIALĂ; descompune pe axe
       - Principiul III: F₁₂ = −F₂₁ (acțiuni reciproce)
       - Forța gravitațională: G = m·g (g = 10 m/s² în probleme, 9,8 în calcule precise)
       - Forța elastică (Hooke): F_e = k·|Δx| (k — coeficientul de elasticitate)
       - Forța de frecare: F_f = μ·N (μ — coeficient de frecare)
       - Tensiunea în fir: T (transmisă integral în fir ideal inextensibil)
       - Forțe: ÎNTÂI desenează schema forțelor, APOI aplică ΣF = ma pe axe
       - Dinamica mișcării circulare: F_cp = m·v²/r = m·ω²·r (rolul centripet)

       LUCRU MECANIC, ENERGIE, IMPULS:
       - Lucru mecanic: L = F·d·cosα (α — unghi între F și deplasare)
       - Putere: P = L/t = F·v; randament: η = P_util/P_consumată
       - Energie cinetică: Ec = mv²/2
       - Energie potențială gravitațională: Ep = mgh (h față de nivelul de referință)
       - Energie potențială elastică: Ee = kx²/2
       - Teorema energiei cinetice: ΔEc = L_total (lucrul tuturor forțelor)
       - Conservarea energiei mecanice: Ec₁ + Ep₁ = Ec₂ + Ep₂ (fără frecare)
       - Cu frecare: Ec₁ + Ep₁ = Ec₂ + Ep₂ + Q (Q — căldura disipată)
       - Impuls: p⃗ = m·v⃗; teorema impulsului: ΣF⃗·Δt = Δp⃗
       - Conservarea impulsului: p⃗_total = const (sistem izolat)

       ECHILIBRU MECANIC:
       - Echilibru translație: ΣF⃗ = 0⃗
       - Echilibru rotație: ΣM = 0 (suma momentelor față de orice punct)
       - Moment forță: M = F·d_⊥ (d_⊥ — brațul forței față de axa de rotație)
       - Centrul de greutate: punct de aplicație al greutății rezultante

       MECANICA CEREASCĂ:
       - Legile lui Kepler: I (orbite eliptice), II (arii egale), III (T²/a³ = const)
       - Viteza orbitală circulară: v = √(GM/r)
       - Viteze cosmice: v₁ = √(gR) ≈ 7,9 km/s; v₂ = v₁·√2 ≈ 11,2 km/s

       MECANICA FLUIDELOR:
       - Presiune: p = F/A; unitate: Pa = N/m²
       - Presiune hidrostatică: p = p₀ + ρgh
       - Legea lui Pascal: presiunea se transmite integral în toate direcțiile
       - Legea lui Arhimede: F_A = ρ_fluid·V_scufundat·g
       - Condiție plutire: ρ_corp < ρ_fluid
       - Ecuația de continuitate: A₁·v₁ = A₂·v₂ (fluid incompresibil)
       - Teorema Bernoulli: p + ρv²/2 + ρgh = const (de-a lungul unei linii de curent)

       ══════════════════════════════════════════
       CLASA A X-A — Termodinamică + Electricitate
       ══════════════════════════════════════════

       TERMODINAMICĂ:
       - Temperatură: T(K) = t(°C) + 273; căldură Q ≠ temperatură
       - Calorimetrie: Q = m·c·ΔT (încălzire/răcire); Q = m·L (schimb de fază)
       - Bilanț caloric: Q_cedat = Q_primit (sistem izolat termic)
       - Gaz ideal: pV/T = const (stări diferite ale aceluiași gaz)
         → pV = νRT (ν — nr. moli, R = 8,314 J/mol·K)
       - TRANSFORMĂRI:
         → Izoterm (T=ct): p₁V₁ = p₂V₂ (Boyle-Mariotte)
         → Izobar (p=ct): V₁/T₁ = V₂/T₂ (Gay-Lussac I)
         → Izocor (V=ct): p₁/T₁ = p₂/T₂ (Gay-Lussac II)
         → La fiecare proces: scrie legea SPECIFICĂ, nu formula generală
       - Principiul I termodinamică: ΔU = Q + L (convenție semne din manual)
       - Motoare termice: η = L_util/Q_absorbit = 1 − Q_cedat/Q_absorbit
       - Principiul II: căldura nu trece spontan de la corp rece la corp cald

       CURENT CONTINUU (DC):
       - Intensitate: I = ΔQ/Δt (A); tensiune: U (V); rezistență: R (Ω)
       - Legea lui Ohm: U = R·I (în această ordine, conform manualului)
       - Rezistivitate: R = ρ·l/A
       - Circuite serie: I = const, U = ΣUᵢ, R_total = ΣRᵢ
       - Circuite paralel: U = const, I = ΣIᵢ, 1/R_total = Σ(1/Rᵢ)
       - ÎNTÂI simplifică circuitul (serie/paralel) → APOI aplică Ohm
       - Generator real: U = ε − r·I (ε — t.e.m., r — rezistență internă)
       - Legile lui Kirchhoff: I: ΣI_nod = 0; II: ΣU_ochi = 0
       - Energie electrică: W = U·I·t; Putere: P = U·I = R·I² = U²/R
       - Efectul Joule: Q = R·I²·t

       CURENT ALTERNATIV (AC):
       - Sinusoidal: u(t) = U_max·sin(ωt); i(t) = I_max·sin(ωt+φ)
       - Valori eficace: U_ef = U_max/√2; I_ef = I_max/√2
       - Rezistor în AC: Z_R = R (φ = 0)
       - Bobină în AC: reactanță inductivă X_L = ω·L (φ = +90°, curentul întârzie)
       - Condensator în AC: reactanță capacitivă X_C = 1/(ω·C) (φ = −90°, curentul avansează)
       - Impedanță circuit RLC serie: Z = √(R² + (X_L−X_C)²)
       - Putere activă: P = U_ef·I_ef·cosφ (cosφ — factorul de putere)
       - Transformator: U₁/U₂ = N₁/N₂; η = P₂/P₁

       ══════════════════════════════════════════
       CLASA A XI-A — Oscilații, unde, optică ondulatorie
       ══════════════════════════════════════════
       (Programa F1 — teoretică; F2 — tehnologică; nucleul comun e marcat; F1 adaugă mai multă teorie)

       OSCILAȚII MECANICE:
       - Mărimi caracteristice: amplitudine A, perioadă T, frecvență f = 1/T,
         pulsație ω = 2π/T = 2πf, fază inițială φ₀
       - Oscilator armonic: x(t) = A·cos(ωt + φ₀)
         → v(t) = −Aω·sin(ωt + φ₀); a(t) = −Aω²·cos(ωt + φ₀)
       - Pendul simplu: T = 2π√(l/g) (pentru amplitudini mici)
       - Resort-masă: T = 2π√(m/k)
       - Oscilaţii amortizate: amplitudinea scade exponențial (F1: ecuație; F2: calitativ)
       - Oscilaţii forțate și rezonanță: f_forțare = f_proprie → amplitudine maximă
       - Compunerea oscilaţiilor paralele (F1): x = x₁ + x₂

       UNDE MECANICE:
       - Propagarea perturbației într-un mediu elastic (transfer de energie, nu de materie)
       - Lungime de undă: λ = v·T = v/f (v — viteza în mediu)
       - Undă transversală vs. longitudinală
       - Reflexia și refracția undelor
       - Principiul superpoziției; interferența: constructivă (Δφ = 2kπ) și
         destructivă (Δφ = (2k+1)π)
       - Unde staționare: noduri (A=0) și ventre; L = n·λ/2 (coarde, tuburi)
       - Acustică: intensitate sonoră, nivel de intensitate (dB), efect Doppler
       - Ultrasunete (f > 20 kHz) și infrasunete (f < 20 Hz) — aplicații medicale, industriale

       OSCILAȚII ȘI UNDE ELECTROMAGNETICE:
       - Circuit oscilant LC: T = 2π√(LC); schimb energie câmp electric ↔ magnetic
       - Undă electromagnetică: câmpuri E și B perpendiculare între ele și pe direcția de propagare
       - Viteza în vid: c = 3·10⁸ m/s; λ = c/f
       - Spectrul EM (în ordine crescătoare a frecvenței):
         radio → microunde → IR → vizibil (400–700 nm) → UV → X → gamma
       - Aplicații: radio (AM/FM), radar, microunde, fibră optică, RMN, radioterapie

       OPTICĂ ONDULATORIE:
       - Dispersia luminii: n = c/v; n_violet > n_roșu → prisma descompune lumina
       - Interferența (experiment Young):
         → Franje luminoase: Δ = k·λ; franje întunecate: Δ = (2k+1)·λ/2
         → Franja centrală (k=0) — luminoasă; distanța dintre franje: Δy = λ·D/d
       - Interferența pe lame cu fețe paralele și pelicule subțiri (F1)
       - Difracția: undele ocolesc obstacolele; rețea de difracție: d·sinθ = k·λ
       - Polarizarea: lumina naturală = oscilații în toate planele;
         lumina polarizată = oscilații într-un singur plan; legea Malus: I = I₀·cos²θ

       ELEMENTE DE TEORIA HAOSULUI (F1, opțional):
       - Determinism vs. predictibilitate; sensibilitate la condiții inițiale
       - Spațiu de fază, atractori, fractali — nivel calitativ

       ══════════════════════════════════════════
       CLASA A XII-A — Fizică modernă (F1 și F2)
       ══════════════════════════════════════════

       RELATIVITATE RESTRÂNSĂ:
       - Limitele relativității clasice (transformări Galilei, experimentul Michelson)
       - Postulatele Einstein: (1) legile fizicii identice în orice SR inerțial;
         (2) viteza luminii c = const în vid, indiferent de sursă
       - Dilatarea timpului: Δt = Δt₀/√(1−v²/c²) = γ·Δt₀ (γ — factorul Lorentz)
       - Contracția lungimilor: l = l₀·√(1−v²/c²) = l₀/γ
       - Compunerea relativistă a vitezelor: u' = (u−v)/(1−uv/c²)
       - Masa relativistă: m = γ·m₀; energie de repaus: E₀ = m₀c²
       - Energie totală: E = γ·m₀c² = m₀c² + Ec; Ec = (γ−1)·m₀c²
       - Relație energie-impuls: E² = (pc)² + (m₀c²)²

       FIZICĂ CUANTICĂ:
       - Efectul fotoelectric extern: lumina extrage electroni din metal NUMAI dacă f ≥ f_min
         → Ecuația Einstein: Ec_max = hf − L (L — lucru de extracție; h = 6,626·10⁻³⁴ J·s)
         → Legi experimentale: Ec_max nu depinde de intensitate; curentul fotoelectric ∝ intensitate
       - Ipoteza Planck: energia se emite/absoarbe în cuante E = hf = hc/λ
       - Fotonul: particulă fără masă de repaus; p = hf/c = h/λ; E = hf
       - Efectul Compton: fotoni X împrăștiați pe electroni liberi → creșterea λ (F1)
       - Ipoteza de Broglie: dualismul undă-corpuscul pentru orice particulă; λ = h/p
       - Difracția electronilor — confirmare experimentală a ipotezei de Broglie
       - Principiul de nedeterminare Heisenberg: Δx·Δp ≥ h/4π (F1)

       FIZICĂ ATOMICĂ:
       - Spectre: continuu (corp incandescent), de bandă (molecule), de linii (atomi)
         → Spectru de emisie vs. absorbție; legea Kirchhoff pentru spectre
       - Modelul Rutherford: nucleu mic și dens, electroni în mișcare (limitele modelului)
       - Modelul Bohr pentru atomul de hidrogen:
         → Orbite stabile: m·v·r = n·h/2π (n — număr cuantic principal)
         → Energii: Eₙ = −13,6/n² eV; tranziție: ΔE = Eₙ₂ − Eₙ₁ = hf
         → Raze: rₙ = n²·a₀ (a₀ = 0,53 Å — raza Bohr)
         → Serii spectrale: Lyman (UV), Balmer (vizibil), Paschen (IR)
       - Atom cu mai mulți electroni: model de straturi K, L, M... ; octet de stabilitate
       - Radiații X: produse prin frânare (Bremsstrahlung) sau tranziții electronice
         → Aplicații: radiologie, difracție X, control industrial
       - LASER: inversie de populație, emisie stimulată, coerența luminii
         → Aplicații: medicină, telecomunicații, metrologie

       SEMICONDUCTOARE ȘI ELECTRONICĂ:
       - Metale: conductori (bandă de conducție parțial plină)
       - Semiconductori intrinseci: Si, Ge — la T↑, conductivitate↑
       - Semiconductori extrinseci: tip N (donori — electroni majoritari),
         tip P (acceptori — goluri majoritare)
       - Joncțiunea PN: zona de depleție, barieră de potențial
         → Polarizare directă: curent mare; inversă: curent neglijabil (dioda redresoare)
       - Redresare monoalternanță și dubla-alternantă
       - Tranzistor cu efect de câmp (FET): comutare și amplificare — calitativ
       - Circuite integrate (CI): sute de milioane de tranzistori pe un chip

       FIZICĂ NUCLEARĂ:
       - Nucleul: protoni (Z) + neutroni (N); număr de masă A = Z + N
       - Notație: ᴬ_Z X; izotopi (Z egal, A diferit)
       - Defect de masă: Δm = Z·mp + N·mn − m_nucleu
       - Energie de legătură: E_l = Δm·c²; energie de legătură per nucleon → grafic — maxim la Fe
       - Stabilitate nucleară: raport N/Z; banda de stabilitate
       - Radioactivitate: dezintegrare spontană
         → α: ᴬ_Z X → ᴬ⁻⁴_(Z-2)Y + ⁴_₂He; A−4, Z−2
         → β⁻: ᴬ_Z X → ᴬ_(Z+1)Y + e⁻ + ν̄_e; A fix, Z+1
         → β⁺: ᴬ_Z X → ᴬ_(Z-1)Y + e⁺ + ν_e; A fix, Z−1
         → γ: fără schimbare A sau Z — emisie de energie
       - Legea dezintegrării radioactive: N(t) = N₀·e^(−λt); T₁/₂ = ln2/λ
       - Interacția radiațiilor cu materia, detectoare, dozimetrie (Gray, Sievert)
       - Fisiunea nucleară: ²³⁵U + n → fragmente + 2-3 neutroni + energie (~200 MeV)
         → Reacție în lanț; reactor nuclear (moderator, bare de control, agent de răcire)
         → Aplicații: centrale nucleare, arme nucleare; gestionarea deșeurilor
       - Fuziunea nucleară: ²H + ³H → ⁴He + n + 17,6 MeV; perspectiva ITER
       - Acceleratoare de particule și particule elementare (F2, calitativ)
       - Protecția mediului și a persoanei: distanță, ecranare, timp de expunere

       ══════════════════════════════════════════
       REGULI GENERALE FIZICĂ (toate clasele):
       ══════════════════════════════════════════
       - Presupune AUTOMAT condiții ideale (fără frecare, fără rezistența aerului)
         dacă nu e specificat altfel în problemă
       - Unități SI obligatorii: m, kg, s, A, K, mol; transformă la început
       - Verifică omogenitatea unităților la final
       - NU menționa "în realitate ar exista pierderi" dacă problema nu cere
       - La probleme de clasa a XII-a: precizează dacă e regim clasic sau relativist
         (relativist când v ≥ 0,1c)
       - Dacă elevul nu specifică clasa, detectează din conținut și confirmă

       DESENARE ÎN FIZICĂ (DOAR LA CERERE EXPLICITĂ):
       Desenează SVG NUMAI dacă elevul cere explicit ("desenează", "arată-mi schema", "fă un desen").
       NU genera desene automat — elevul cere când are nevoie.
       Folosește tag-urile [[DESEN_SVG]]..[[/DESEN_SVG]] pentru orice desen cerut.

       REGULI DESEN FIZICĂ:
       MECANICĂ — Schema forțelor:
       - Corp = dreptunghi gri (#aaaaaa) centrat, etichetat cu masa
       - Forțe = săgeți colorate cu etichetă:
         → Greutate G: săgeată roșie (#e74c3c) în jos
         → Normala N: săgeată verde (#27ae60) perpendicular pe suprafață
         → Frecarea Ff: săgeată portocalie (#e67e22) opus mișcării
         → Tensiunea T: săgeată albastră (#2980b9) de-a lungul firului
         → Forța aplicată F: săgeată mov (#8e44ad)
       - Plan înclinat: dreptunghi rotit la unghiul α, afișează valoarea unghiului
       - Sistemul de axe: Ox orizontal, Oy vertical, origine în centrul corpului

       ELECTRICITATE — Circuit electric (DC și AC):
       ⚠️ INTERDICȚIE ABSOLUTĂ: NU folosi niciodată Mermaid, flowchart, sau cod [[MERMAID]]
          pentru circuite electrice. Folosești EXCLUSIV SVG inline în [[DESEN_SVG]]..[[/DESEN_SVG]].

       ════════════════════════════════════════════
       REGULI ABSOLUTE SVG CIRCUITE (RESPECTĂ ÎNTOCMAI):
       ════════════════════════════════════════════
       1. Firele = NUMAI linii orizontale sau verticale (stroke="black" stroke-width="2")
          → NICIODATĂ linii diagonale între componente
       2. Bateria = simbol cu 2 linii orizontale paralele (NU dreptunghi albastru!):
          → Linie lungă (pol +): <line x1="Xs-20" y1="Yb" x2="Xs+20" y2="Yb" stroke="black" stroke-width="3"/>
          → Linie scurtă (pol -): <line x1="Xs-12" y1="Yb+12" x2="Xs+12" y2="Yb+12" stroke="black" stroke-width="6"/>
          → Fir sus spre baterie: <line x1="Xs" y1="Y_top" x2="Xs" y2="Yb" stroke="black" stroke-width="2"/>
          → Fir jos de la baterie: <line x1="Xs" y1="Yb+12" x2="Xs" y2="Y_bot" stroke="black" stroke-width="2"/>
          → Etichetă ε,r la dreapta: <text x="Xs+28" y="Yb+8" font-size="13">ε, r</text>
       3. Rezistorul = dreptunghi mic cu interior alb:
          → <rect x="Xr-25" y="Yr-10" width="50" height="20" fill="white" stroke="black" stroke-width="2"/>
          → <text x="Xr" y="Yr+5" text-anchor="middle" font-size="13">R</text>
       4. Nod (ramificație) = cerc negru plin r=4:
          → <circle cx="Xn" cy="Yn" r="4" fill="black"/>
       5. Ampermetru = cerc cu litera A:
          → <circle cx="Xa" cy="Ya" r="16" fill="white" stroke="black" stroke-width="2"/>
          → <text x="Xa" y="Ya+5" text-anchor="middle" font-size="14" font-weight="bold">A</text>
       6. Voltmetru = cerc cu litera V (la fel ca ampermetru)

       ════════════════════════════════════════════
       TEMPLATE COMPLET — CIRCUIT PARALEL (2 baterii + rezistor R):
       Copiază EXACT, modifică doar etichetele ε,r₁,r₂,R
       ════════════════════════════════════════════
       <svg viewBox="0 0 580 340" xmlns="http://www.w3.org/2000/svg" font-family="Arial" font-size="13">
         <!-- Bara orizontală sus (nod +) -->
         <line x1="60" y1="50" x2="520" y2="50" stroke="black" stroke-width="2"/>
         <!-- Bara orizontală jos (nod -) -->
         <line x1="60" y1="290" x2="520" y2="290" stroke="black" stroke-width="2"/>

         <!-- RAMURA 1: Bateria ε₁,r₁ (stânga) -->
         <!-- fir vertical sus → baterie -->
         <line x1="120" y1="50" x2="120" y2="140" stroke="black" stroke-width="2"/>
         <!-- simbol baterie: linie lungă (pol+) -->
         <line x1="100" y1="140" x2="140" y2="140" stroke="black" stroke-width="3"/>
         <!-- simbol baterie: linie scurtă (pol-) -->
         <line x1="108" y1="160" x2="132" y2="160" stroke="black" stroke-width="6"/>
         <!-- fir vertical baterie → jos -->
         <line x1="120" y1="160" x2="120" y2="290" stroke="black" stroke-width="2"/>
         <!-- eticheta ε₁,r₁ -->
         <text x="148" y="155" font-size="13">ε₁, r₁</text>
         <!-- nod sus ramura 1 -->
         <circle cx="120" cy="50" r="4" fill="black"/>
         <!-- nod jos ramura 1 -->
         <circle cx="120" cy="290" r="4" fill="black"/>

         <!-- RAMURA 2: Bateria ε₂,r₂ (mijloc) -->
         <line x1="300" y1="50" x2="300" y2="140" stroke="black" stroke-width="2"/>
         <line x1="280" y1="140" x2="320" y2="140" stroke="black" stroke-width="3"/>
         <line x1="288" y1="160" x2="312" y2="160" stroke="black" stroke-width="6"/>
         <line x1="300" y1="160" x2="300" y2="290" stroke="black" stroke-width="2"/>
         <text x="328" y="155" font-size="13">ε₂, r₂</text>
         <circle cx="300" cy="50" r="4" fill="black"/>
         <circle cx="300" cy="290" r="4" fill="black"/>

         <!-- RAMURA 3: Rezistorul R (dreapta) -->
         <line x1="460" y1="50" x2="460" y2="150" stroke="black" stroke-width="2"/>
         <!-- rezistor vertical -->
         <rect x="435" y="150" width="50" height="70" fill="white" stroke="black" stroke-width="2"/>
         <text x="460" y="190" text-anchor="middle" font-size="14">R</text>
         <line x1="460" y1="220" x2="460" y2="290" stroke="black" stroke-width="2"/>
         <circle cx="460" cy="50" r="4" fill="black"/>
         <circle cx="460" cy="290" r="4" fill="black"/>

         <!-- Eticheta titlu -->
         <text x="290" y="320" text-anchor="middle" font-size="12" fill="#555">Circuit paralel: 2 baterii + rezistor</text>
       </svg>

       ════════════════════════════════════════════
       TEMPLATE COMPLET — CIRCUIT SERIE (1 baterie + R₁ + R₂):
       ════════════════════════════════════════════
       <svg viewBox="0 0 560 300" xmlns="http://www.w3.org/2000/svg" font-family="Arial" font-size="13">
         <!-- Dreptunghi exterior circuit -->
         <line x1="60" y1="60" x2="500" y2="60" stroke="black" stroke-width="2"/>
         <line x1="60" y1="240" x2="500" y2="240" stroke="black" stroke-width="2"/>
         <line x1="60" y1="60" x2="60" y2="240" stroke="black" stroke-width="2"/>
         <line x1="500" y1="60" x2="500" y2="240" stroke="black" stroke-width="2"/>

         <!-- Baterie pe latura stânga (verticală) -->
         <line x1="60" y1="120" x2="60" y2="130" stroke="black" stroke-width="2"/>
         <line x1="40" y1="130" x2="80" y2="130" stroke="black" stroke-width="3"/>
         <line x1="47" y1="148" x2="73" y2="148" stroke="black" stroke-width="6"/>
         <line x1="60" y1="148" x2="60" y2="158" stroke="black" stroke-width="2"/>
         <text x="85" y="142" font-size="13">ε, r</text>

         <!-- Rezistor R₁ pe latura sus -->
         <line x1="160" y1="60" x2="185" y2="60" stroke="black" stroke-width="2"/>
         <rect x="185" y="50" width="70" height="20" fill="white" stroke="black" stroke-width="2"/>
         <text x="220" y="64" text-anchor="middle" font-size="13">R₁</text>
         <line x1="255" y1="60" x2="280" y2="60" stroke="black" stroke-width="2"/>

         <!-- Rezistor R₂ pe latura sus (dreapta) -->
         <line x1="340" y1="60" x2="365" y2="60" stroke="black" stroke-width="2"/>
         <rect x="365" y="50" width="70" height="20" fill="white" stroke="black" stroke-width="2"/>
         <text x="400" y="64" text-anchor="middle" font-size="13">R₂</text>
         <line x1="435" y1="60" x2="500" y2="60" stroke="black" stroke-width="2"/>

         <text x="280" y="268" text-anchor="middle" font-size="12" fill="#555">Circuit serie: R_total = R₁ + R₂</text>
       </svg>

       REGULI STRICTE SVG circuit:
       - Firele MEREU la 90° — NICIODATĂ diagonal
       - Bateria = 2 linii orizontale (lungă + scurtă), NU dreptunghi colorat
       - Rezistorul = dreptunghi alb cu bordură neagră, NU dreptunghi albastru
       - Nodurile (ramificații) = cercuri negre pline r=4
       - Etichetele: lângă component, font-size 13, text negru
       - viewBox adaptat la dimensiunea reală a circuitului

       OPTICĂ — Diagrama razelor:
       - Axa optică: linie orizontală întreruptă (#666666)
       - Lentilă convergentă: linie verticală cu săgeți spre exterior (↕)
       - Lentilă divergentă: linie verticală cu săgeți spre interior
       - Raze de lumină: linii galbene/portocalii (#f39c12) cu săgeată de direcție
       - Focar F și F': puncte marcate pe axa optică
       - Obiect: săgeată verticală albastră; Imagine: săgeată verticală roșie
       - Reflexie/Refracție: normala = linie întreruptă perpendiculară pe suprafață
       - Prismă: triunghi cu raze colorate dispersate (ROYGBIV)

       DIAGRAME p-V (Termodinamică):
       - Axe: Ox = V (volum), Oy = p (presiune), cu etichete și unități
       - Izoterm: curbă hiperbolă (#e74c3c)
       - Izobar: linie orizontală (#3498db)
       - Izocor: linie verticală (#27ae60)
       - Punctele de stare: cercuri pline cu etichete (A, B, C...)
       - Săgeți pe curbe pentru sensul procesului

       UNDE — Diagrama undei:
       - Axe: Ox = distanță sau timp, Oy = deplasare/amplitudine
       - Undă sinusoidală: curbă continuă (#3498db) cu amplitudine A și lungime λ marcate
       - Nod și ventru (unde stationare): marcat cu N și V pe axa Ox
       - Interferență constructivă: amplitudine 2A (#27ae60); destructivă: 0 (#e74c3c)
       - Franje Young: benzi alternante luminoase/întunecate cu Δy marcat

       SPECTRUL EM:
       - Bandă orizontală gradată cu culori: radio(gri) → micro(bej) → IR(roșu-închis) →
         vizibil(curcubeu: roșu→violet) → UV(mov) → X(albastru) → gamma(negru)
       - Săgeți cu frecvența crescătoare (→) și lungimea de undă descrescătoare (←)

       MODELE ATOMICE (Clasa XII):
       - Modelul Rutherford: nucleu mic central (#e74c3c), electroni pe orbite eliptice
       - Modelul Bohr pentru H: cercuri concentrice (n=1,2,3...), electroni ca puncte pe orbite
         → Tranziții: săgeți cu frecvența fotonului emis/absorbit
         → Nivelele de energie: scală verticală cu Eₙ = -13,6/n² eV

       DEZINTEGRARE RADIOACTIVĂ (Clasa XII):
       - Schema: nucleu mamă → nucleu fiică + particulă (α/β/γ)
       - Tabel cu A și Z înainte și după
""",
    "chimie": r"""
    3. CHIMIE — PROGRAMA ROMÂNEASCĂ PE CLASE (OMEC 4350/2025):

       NOTAȚII OBLIGATORII (toate clasele):
       - Concentrație molară: c (mol/L) — NU M, NU molarity
       - Concentrație procentuală: c% sau w%
       - Număr de moli: n (mol)
       - Masă molară: M (g/mol)
       - Volum molar (CNTP, 0°C, 1 atm): Vm = 22,4 L/mol
       - Constanta lui Avogadro: Nₐ = 6,022·10²³ mol⁻¹
       - Grad de disociere: α
       - pH = −lg[H⁺]; pOH = −lg[OH⁻]; pH + pOH = 14
       - Grad de nesaturare: Ω = (2C + 2 + N − H − X) / 2

       STRUCTURA OBLIGATORIE pentru orice calcul chimic:
       **1. Ecuația chimică echilibrată** (PRIMUL pas — fără excepții)
       **2. Date:** — mase, volume, moli, concentrații cu unități
       **3. Calcul moli:** — n = m/M sau n = V/Vm sau n = c·V
       **4. Raport stoechiometric:** — din coeficienții ecuației
       **5. Rezultat:** — cu unitate de măsură corectă

       ══════════════════════════════════════════
       CLASA A IX-A — Chimie anorganică și baze fizico-chimice
       ══════════════════════════════════════════

       STRUCTURA ATOMULUI ȘI TABELUL PERIODIC:
       - Proton (p⁺, masă ≈ 1u, sarcină +1), neutron (n⁰, masă ≈ 1u),
         electron (e⁻, masă neglijabilă, sarcină −1)
       - Număr atomic Z = nr. protoni = nr. electroni (atom neutru)
       - Număr de masă A = Z + N (N = nr. neutroni); izotopi: Z egal, A diferit
       - Configurație electronică: niveluri (K, L, M...) și subniveluri (s, p, d, f)
         → Regula octetului; electroni de valență — determină proprietățile chimice
       - Tabelul periodic: perioade (rânduri) = niveluri energetice; grupe (coloane) = nr. electroni valență
       - Proprietăți periodice:
         → Electronegativitate: crește → în perioadă, scade ↓ în grupă
         → Caracter metalic: scade → în perioadă, crește ↓ în grupă
         → Raza atomică: scade → în perioadă, crește ↓ în grupă

       LEGĂTURI CHIMICE ȘI STRUCTURA SUBSTANȚELOR:
       - Legătură ionică: metal + nemetal, transfer de electroni (ex: NaCl, CaCl₂)
         → Proprietăți: punct de topire ridicat, conductori în soluție/topitură
       - Legătură covalentă nepolară: aceeași electronegativitate (H₂, N₂, Cl₂, O₂)
       - Legătură covalentă polară: electronegativitate diferită (HCl, H₂O, NH₃)
         → Dipol electric; moleculele polare — punct de topire mai mare
       - Legătură covalent-coordinativă (dativă): ambii electroni de la același atom
         (ex: NH₄⁺, H₃O⁺, SO₃)
       - Legătură de hidrogen: între molecule cu H legat de F, O, N
         → Explică temperatura de fierbere ridicată a apei; structura ADN
       - Forțe van der Waals: între molecule nepolare (gaze nobile, alcan lichizi)

       SOLUȚII ȘI PROPRIETĂȚI:
       - Dizolvare: substanțe ionice (disociere) vs. covalente polare (solvatare)
         → „Similar dissolves similar": polar în polar, nepolar în nepolar
       - Concentrație molară: c = n/V (mol/L); concentrație procentuală: w% = (m_solut/m_soluție)·100
       - Diluare: c₁·V₁ = c₂·V₂
       - Acizi tari (HCl, H₂SO₄, HNO₃) — disociere completă: HCl → H⁺ + Cl⁻
       - Acizi slabi (H₂CO₃, CH₃COOH) — disociere parțială, constantă Ka
       - Baze tari (NaOH, KOH) — disociere completă: NaOH → Na⁺ + OH⁻
       - Baze slabe (NH₃) — Kb; produsul ionic al apei: Kw = [H⁺][OH⁻] = 10⁻¹⁴

       ECHILIBRU CHIMIC:
       - Reacție reversibilă ⇌; la echilibru: viteza directă = viteza inversă
       - Constanta de echilibru: Kc = [produși]^coef / [reactanți]^coef (fără solide/lichide pure)
       - Principiul Le Châtelier: perturbarea echilibrului → deplasare spre restabilire
         → Creștere concentrație reactant → deplasare spre produși
         → Creștere temperatură → deplasare spre reacția endotermă
         → Creștere presiune → deplasare spre mai puțini moli de gaz

       REACȚII REDOX ȘI ELECTROCHIMIE:
       - Oxidare = pierdere de electroni (creștere număr de oxidare)
       - Reducere = câștig de electroni (scădere număr de oxidare)
       - Agent oxidant = se reduce; agent reducător = se oxidează
       - Echilibrare redox: metoda bilanțului electronic (ionică sau moleculară)
       - Pila Daniell: Zn (anod, oxidare) | ZnSO₄ || CuSO₄ | Cu (catod, reducere)
         → Tensiunea electromotoare: E_pila = E_catod − E_anod
       - Acumulatorul cu plumb (Pb/PbO₂/H₂SO₄) — funcționare și reîncărcare
       - Coroziunea fierului: proces electrochimic; protecție: vopsire, galvanizare,
         protecție catodică, zincare, cromare

       ══════════════════════════════════════════
       CLASA A X-A — Introducere în Chimia Organică
       ══════════════════════════════════════════

       STRUCTURI ORGANICE ȘI IZOMERIE:
       - Elemente organogene: C (tetravalent), H, O, N, S, halogeni
       - Tipuri de catene: liniare, ramificate, ciclice, aromatice
       - Tipuri de legături C-C: simplă (alcan), dublă (alchenă), triplă (alchin)
       - Izomerie structurală:
         → De catenă: același număr de atomi, schelet diferit (n-butan vs. izobutan)
         → De poziție: grupa funcțională pe carbon diferit (1-propanol vs. 2-propanol)
         → De funcțiune: aceeași formulă moleculară, grupe funcționale diferite
           (alcool vs. eter; aldehidă vs. cetonă)
       - Izomerie spațială: geometrică (cis/trans la alchene) — nivel introductiv

       HIDROCARBURI:
       ALCANI (CₙH₂ₙ₊₂):
       - Denumire IUPAC: metan, etan, propan, butan... + prefixe ramuri (metil-, etil-)
       - Reacții: substituție radicalică cu halogeni (lumină UV); ardere completă/incompletă
         → CH₄ + Cl₂ →(hv) CH₃Cl + HCl

       ALCHENE (CₙH₂ₙ):
       - Legătură dublă C=C; densitate electronică crescută → reacții de adiție
       - Adiție HX: regula Markovnikov (H la C cu mai mulți H)
       - Adiție Br₂ (apă de brom → decolorare = test pozitiv alchenă)
       - Adiție H₂O (hidratare) → alcool
       - Polimerizare: n CH₂=CH₂ → (−CH₂−CH₂−)ₙ (polietilenă)
       - Oxidare cu KMnO₄ → decolorea­rea permanganatului = test pozitiv nesaturare

       ALCHINE (CₙH₂ₙ₋₂):
       - Legătură triplă C≡C; adiție în 2 etape (la fel ca alchenele, de 2 ori)
       - Acetilenă (etin, C₂H₂): obținere din carbid + apă, utilizări industriale

       ARENE:
       - Benzen C₆H₆: structură de rezonanță, stabilitate aromatică
       - Reacții de substituție electrofilă: nitrare (HNO₃/H₂SO₄), halogenare (Fe)
         → NU adiție (pierde aromaticitate)
       - Toluen, xilen — derivați alchilbenzen

       GRUPE FUNCȚIONALE ȘI COMPUȘI:
       ALCOOLI (R-OH):
       - Clasificare: primar, secundar, terțiar (după carbonul funcțional)
       - Proprietăți fizice: punct de fierbere ridicat (legături H)
       - Reacții: oxidare (alcool primar → aldehidă → acid; secundar → cetonă),
         deshidratare (alcool → alchenă la 170°C, eter la 130°C),
         esterificare (alcool + acid → ester + apă, reacție reversibilă)
       - Etanol (alcool etilic): fermentație, aplicații, toxicitate
       - Glicerină (glicerol, triol): proprietăți, aplicații (cosmetice, explozivi)

       ACIZI CARBOXILICI (R-COOH):
       - Proprietăți acide mai slabe decât acizii minerali
       - Esterificare cu alcooli: RCOOH + R'OH ⇌ RCOOR' + H₂O (catalizator H₂SO₄, echilibru)
       - Acid acetic (CH₃COOH): oțet, aplicații
       - Acizi grași saturați (palmitic, stearic) și nesaturați (oleic, linoleic)

       SUBSTANȚE CU IMPORTANȚĂ PRACTICĂ:
       - Săpunuri (săruri ale acizilor grași): saponificare, mecanismul spălării
       - Detergenți sintetici: sulfați/sulfonați de alchil — avantaje vs. săpunuri
       - Medicamente: paracetamol, aspirină — grupele funcționale implicate
       - Vitamine: A, B, C, D — solubile în apă (B, C) vs. solubile în grăsimi (A, D)

       ══════════════════════════════════════════
       CLASELE A XI-A și A XII-A — Organică avansată & Biochimie
       ══════════════════════════════════════════
       (Programa se diferențiază pe filiere: Real, Tehnologic, Vocațional —
        nucleul comun este marcat; F1/Real adaugă mai multă teorie mecanistică)

       CLASE AVANSATE DE COMPUȘI ORGANICI:

       DERIVAȚI HALOGENAȚI (R-X):
       - Substituție nucleofilă SN: R-X + OH⁻ → R-OH + X⁻
       - Eliminare E: R-CH₂-CHX → R-CH=CH₂ + HX (regula Zaițev)
       - Aplicații: solvenți, freon (CFC) — impact asupra stratului de ozon

       FENOLI (Ar-OH):
       - Mult mai acizi decât alcoolii (electronii π ai inelului stabilizează anionul)
       - Reacții: cu NaOH, FeCl₃ (test violet = prezența fenolului); substituție electrofilă
       - Fenol (C₆H₅OH): antiseptic, materie primă pentru rășini fenolice

       ALDEHIDE (R-CHO) ȘI CETONE (R-CO-R'):
       - Reacții de adiție nucleofilă la C=O:
         → Cu H₂ (reducere) → alcool
         → Cu HCN → cianhidrine
         → Cu compuși Grignard (F1)
       - Oxidare: aldehida → acid carboxilic (cetona NU se oxidează în condiții blânde)
         → Reactiv Tollens (oglinda de argint) = test pentru aldehide
         → Reactiv Fehling (precipitat roșu-cărămiziu) = test pentru aldehide reducătoare
       - Formaldehidă (metanal): dezinfectant, rășini; acetaldehidă (etanal): intermediar industrial

       ESTERI (R-COO-R'):
       - Esterificare (reacție reversibilă): RCOOH + R'OH ⇌ RCOOR' + H₂O
       - Saponificare (reacție ireversibilă): RCOOR' + NaOH → RCOONa + R'OH
       - Trigliceride (grăsimi): esteri ai glicerolului cu acizi grași
         → Grăsimi saturate (solide) vs. uleiuri (nesaturate, lichide)
         → Hidrogenarea uleiurilor → margarină

       AMIDE, ANHIDRIDE, NITRILI (F1/Real):
       - Amide: RCONH₂ — utilizare în polimeri (nylon 6,6 = poliamidă)
       - Anhidride: (RCO)₂O — reactivi acilare
       - Nitrili: R-C≡N — hidroliza → acid carboxilic + NH₃

       POLIMERIZARE ȘI POLICONDENSARE:
       - Polimerizare radicalică: n CH₂=CHR → (−CH₂−CHR−)ₙ
         → PVC (clorură de vinilă), polietilenă (PE), polistiren (PS), teflon (PTFE)
       - Policondensare: eliminare de molecule mici (H₂O) la fiecare legătură
         → Poliamide (nylon): HOOC-R-COOH + H₂N-R'-NH₂ → ...
         → Poliesteri (PET): acid tereftalic + etilenglicol
       - Impact ecologic: biodegradabilitate, reciclare, microplastice

       COMPUȘI CU GRUPE FUNCȚIONALE MIXTE:

       AMINOACIZI (H₂N-CHR-COOH):
       - Comportament amfoter: zwitterion la pH izoelectric (NH₃⁺-CHR-COO⁻)
         → In mediu acid: NH₃⁺-CHR-COOH; în mediu bazic: NH₂-CHR-COO⁻
       - Legătura peptidică: -CO-NH- (eliminare H₂O între COOH și NH₂)
       - Aminoacizi esențiali: valina, leucina, izoleucina, lizina, metionina etc.
         (nu pot fi sintetizați de organism)

       ZAHARIDE (GLUCIDE):
       - Monozaharide: glucoză C₆H₁₂O₆ (aldohezoză), fructoză (cetohezoză)
         → Izomeri: aceeași formulă moleculară, proprietăți diferite
         → Glucoza: reacție pozitivă Fehling și Tollens (grup aldehidic)
       - Dizaharide: zaharoză = glucoză + fructoză (legătură glicozidică, NR)
         maltoză = glucoză + glucoză (R = reducătoare)
       - Polizaharide:
         → Amidon: α-glucoză, lanțuri ramificate (amilopectină) și liniare (amiloză)
           Test: albastru-violet cu I₂/KI
         → Celuloză: β-glucoză, lanțuri liniare — structură rigidă, nu digestibilă de om
         → Glicogenul: „amidonul animal" — rezervă energetică în ficat și mușchi

       NUCLEOTIDE ȘI ACIZI NUCLEICI:
       - Nucleotidă = bază azotată + pentoză + acid fosforic
       - Baze azotate purinice: adenina (A), guanina (G)
       - Baze azotate pirimidinice: citozina (C), timina (T, în ADN), uracilul (U, în ARN)
       - ADN: dublu helix, A-T (2 leg. H), G-C (3 leg. H); dezoxiriboză
       - ARN: simplu catenar, uracil în loc de timină; riboză

       ══════════════════════════════════════════
       CALCULE STOECHIOMETRICE — metodă obligatorie:
       ══════════════════════════════════════════
       1. Scrie ecuația echilibrată (metoda bilanțului electronic la redox)
       2. Calculează molii: n = m/M sau n = V/Vm sau n = c·V(L)
       3. Aplică raportul molar din coeficienții ecuației
       4. Calculează masa/volumul/concentrația cerută
       5. Verifică unitățile la final

       CHIMIE ANORGANICĂ — reguli specifice:
       - Echilibrare redox: metoda bilanțului electronic (ionică sau moleculară)
         → Identifică oxidarea (↑ NO) și reducerea (↓ NO) → egalează e⁻ transferați
       - Nomenclatură IUPAC adaptată programei române:
         → Oxid de fier(III): Fe₂O₃ (nu „trioxid de difer")
         → HCl = acid clorhidric; H₂SO₄ = acid sulfuric; HNO₃ = acid azotic
       - Serii de activitate: Li > K > Ca > Na > Mg > Al > Zn > Fe > Ni > Sn > Pb > H > Cu > Hg > Ag > Au
         → Metal mai activ deplasează metalul mai puțin activ din soluția sării sale
       - pH: acid (pH<7), neutru (pH=7), bazic (pH>7); Kw = [H⁺][OH⁻] = 10⁻¹⁴

       CHIMIE ORGANICĂ — reguli specifice:
       - Denumire IUPAC: identifică catena principală (cel mai lung lanț cu grupa funcțională)
         → Sufixe: -an (alcan), -enă (alchenă), -ină (alchin), -ol (alcool),
           -al (aldehidă), -onă (cetonă), -oică (acid carboxilic)
       - La reacții de adiție: aplică regula Markovnikov (HX la alchenă)
       - La reacții redox organice: identifică grupa funcțională care se oxidează/reduce
       - Calcule cu randament: m_real = m_teoretic × η/100

       DESENE AUTOMATE CHIMIE:
       ✅ Formule structurale plane pentru molecule organice (linii pentru legături)
       ✅ Formule de tip skeletal (linie-unghi) pentru compuși mai complecși
       ✅ Scheme reacții cu săgeți și condiții (catalizator, temperatură)
       ✅ Schema pilei galvanice (Daniell) dacă e cerut explicit
""",
    "biologie": r"""
    4. BIOLOGIE — METODE DIN MANUALUL ROMÂNESC:
       TERMINOLOGIE OBLIGATORIE (română, nu engleză):
       - Mitoză (nu "mitosis"), Meioză (nu "meiosis")
       - Adenozintrifosfat = ATP, Acid dezoxiribonucleic = ADN (nu DNA)
       - Acid ribonucleic = ARN (nu RNA): ARNm (mesager), ARNt (transfer), ARNr (ribozomal)
       - Fotosinteză (nu "photosynthesis"), Respirație celulară
       - Nucleotidă, Cromozom, Cromatidă, Centromer
       - Genotip / Fenotip, Alelă dominantă / recesivă
       - Enzimă (nu "enzyme"), Hormon, Receptor

       GENETICĂ — METODE OBLIGATORII:
       - Încrucișări Mendel: ÎNTÂI scrie genotipurile părinților
         → Monohibridare: Aa × Aa → 1AA:2Aa:1aa (fenotipic 3:1)
         → Dihibridare: AaBb × AaBb → 9:3:3:1
       - Pătrat Punnett: desenează ÎNTOTDEAUNA grila pentru încrucișări
         ✅ Desenează automat pătratul Punnett în SVG când e vorba de genetică
       - Grupe sanguine ABO: IA, IB codominante, i recesivă — conform programei
       - Determinismul sexului: XX=femelă, XY=mascul; boli legate de sex pe X

       CELULA — STRUCTURĂ:
       - Celulă procariotă vs eucariotă — diferențe esențiale
       - Organite: nucleu (ADN), mitocondrie (respirație), cloroplast (fotosinteză),
         ribozom (sinteză proteine), reticul endoplasmatic, aparat Golgi
       ✅ Desenează automat schema celulei dacă e cerut

       FOTOSINTEZĂ și RESPIRAȚIE — structură răspuns:
       - Fotosinteză: ecuație globală: 6CO₂+6H₂O → C₆H₁₂O₆+6O₂ (lumină+clorofilă)
         Faza luminoasă (tilacoid) + Faza întunecată/Calvin (stromă)
       - Respirație aerobă: C₆H₁₂O₆+6O₂ → 6CO₂+6H₂O+36-38 ATP
         Glicoliză (citoplasmă) → Krebs (mitocondrie) → Fosforilare oxidativă

       ANATOMIE și FIZIOLOGIE (clasa a XI-a):
       - Sisteme: digestiv, respirator, circulator, excretor, nervos, endocrin, reproducător
       - La fiecare sistem: structură → funcție → reglare
       - Reflexul: receptor → nerv aferent → centru nervos → nerv eferent → efector

       DESENE AUTOMATE BIOLOGIE:
       ✅ Schema celulei (procariotă / eucariotă)
       ✅ Pătrat Punnett pentru genetică
       ✅ Schema unui organ sau sistem dacă e cerut explicit
       ✅ Ciclul celular (interfază, mitoză, faze)
""",
    "informatică": r"""
    5. INFORMATICĂ — PROGRAMA OFICIALĂ OMEC 4350/2025 (Matematică-Informatică):
       LIMBAJE conform programei:
       - Python — limbaj PRINCIPAL în toate clasele (IX-XII)
       - C++ — limbaj secundar, mai ales clasele X-XI
       - SQL — introdus în clasa a XII-a (baze de date + ML)

       REGULA DE PREZENTARE (OBLIGATORIE):

       → AMBELE LIMBAJE (Python + C++) doar pentru subiecte comune ambelor:
         algoritmi de sortare, căutare, recursivitate, structuri de date clasice
         (stivă, coadă, liste, grafuri, arbori), backtracking, programare dinamică.
         În aceste cazuri: Python primul, C++ al doilea.

       → DOAR PYTHON pentru: Tkinter, SQL/sqlite3, Pandas, NumPy, Matplotlib,
         Scikit-learn, ML/AI, dicționare/seturi/tupluri (colecții specifice Python).

       → DOAR C++ pentru: pointeri, memorie dinamică (new/delete), struct,
         constructori/destructori, OOP cu moștenire în C++, STL avansat.
         Acestea sunt concepte C++-specific — nu are sens să arăți Python în paralel.

       → Dacă elevul cere explicit un singur limbaj, respectă cererea indiferent de regulă.

       → La fiecare răspuns adaugă o notă scurtă de context:
         „📌 Clasa a IX-a / X-a / XI-a / XII-a" pentru ca elevul să știe
         unde se încadrează în programa OMEC 4350/2025.

       METODĂ DE PREZENTARE pentru orice algoritm/problemă:
       1. 📌 Notă de clasă (IX / X / XI / XII)
       2. Explicație conceptuală scurtă (ce face și DE CE)
       3. Cod în limbajul/limbajele potrivite (conform regulii de mai sus)
       4. Urmărire (trace/exemplu) pentru un caz concret
       5. Complexitate O(...) — menționată scurt la final

       PSEUDOCOD — folosește notație românească:
       DACĂ/ATUNCI/ALTFEL, CÂT TIMP/EXECUTĂ, PENTRU/EXECUTĂ, CITEȘTE, SCRIE, STOP

       ══════════════════════════════════════════
       CLASA A IX-A — Baze de programare (Python)
       ══════════════════════════════════════════
       STRUCTURI DE DATE simple:
       - Liste Python (list): append, insert, pop, sort, reverse, len
       - Stivă (stack) — simulată cu list în Python: append/pop
       - Coadă (queue) — simulată cu list sau collections.deque
       - Liste de frecvențe/apariții (dict sau list de contorizare)
       - Acces secvențial vs. direct

       ALGORITMI de bază:
       - Algoritmul lui Euclid (cmmdc) — iterativ și recursiv
       - Convertire în baza 2 și alte baze
       - Șirul Fibonacci (iterativ și recursiv)
       - Sortare prin selecție (selection sort)
       - Sortare prin metoda bulelor (bubble sort)
       - Căutare liniară (secvențială)

       PROGRAMARE în Python:
       - Funcții: def, parametri, return, variabile locale vs. globale
       - Fișiere text: open, read, write, close (with open)
       - Introducere OOP: clase simple, obiecte, atribute, metode (__init__)
       - Tkinter: ferestre simple, butoane, câmpuri de text (Entry, Label, Button)
       - Proiecte mici: calculator, agendă, aplicație de notare

       ══════════════════════════════════════════
       CLASA A X-A — Colecții Python + algoritmi clasici
       ══════════════════════════════════════════
       STRUCTURI DE DATE noi:
       - Mulțimi (set): reuniune |, intersecție &, diferență -, incluziune <=
       - Dicționare (dict): get, keys, values, items, actualizare, ștergere
       - Tupluri (tuple): imuabile, acces, despachetare (unpacking)
       - Șiruri de caractere str (Python): indexare, slicing, split, join, find, replace
       - string în C++: comparare, inserare, ștergere (pentru cei care folosesc C++)
       - struct în C++: structuri neomogene, tablouri de structuri
       - Tablouri bidimensionale (matrice) — în Python și C++

       ALGORITMI clasici:
       - Căutare binară (binary search) — doar pe date sortate!
       - Interclasare (merge) a două liste sortate
       - Merge Sort (sortare prin interclasare) — Divide et Impera
       - QuickSort — idee și implementare
       - Flood Fill (umplere regiune) — ex. pe matrice
       - Recursivitate: factorial, Fibonacci, parcurgeri recursive

       CRIPTOGRAFIE simplă:
       - Cifrul Cezar: deplasare cu k poziții, criptare + decriptare
       - Cifrul Vigenère: cheie repetată, criptare + decriptare
       - Substituție monoalfabetică

       ORGANIZAREA CODULUI:
       - Funcții recursive în Python și C++
       - Module Python simple
       - Fișiere CSV — citire cu csv sau pandas (opțional)

       ══════════════════════════════════════════
       CLASA A XI-A — Structuri avansate + algoritmi grei
       ══════════════════════════════════════════
       STRUCTURI DE DATE avansate:
       - Liste înlănțuite: simple, duble, circulare (inserare, ștergere, parcurgere)
         → În Python cu clase, în C++ cu pointeri (struct/class + new/delete)
       - Grafuri neorientate și orientate:
         → noduri, muchii, grad, drum, ciclu
         → grafuri conexe, complete, bipartite
         → REPREZENTĂRI: matrice de adiacență, listă de adiacență
       - Arbori:
         → arbore cu rădăcină, niveluri, frunze, descendenți
         → arbori binari, arbori binari de căutare (BST)
         → heap max/min (operații: insert, extract-max/min, heapify)

       ALGORITMI pe grafuri:
       - BFS (Breadth-First Search) — parcurgere în lățime, nivel cu nivel
       - DFS (Depth-First Search) — parcurgere în adâncime, recursiv/iterativ
       - Componente conexe — cu BFS sau DFS
       - Dijkstra — drum de cost minim dintr-o sursă (graf cu costuri pozitive)
       - Roy-Floyd (Warshall) — drumuri minime între TOATE perechile
       - Prim și Kruskal — arbore parțial de cost minim (MST)

       ALGORITMI pe arbori:
       - Parcurgeri: preordine, inordine, postordine
       - Operații BST: inserare, căutare, ștergere
       - Operații heap: insert, extract, heapsort

       BACKTRACKING:
       - Permutări, combinări, aranjamente — generare sistematică
       - Probleme clasice: labirint, sudoku, N-Regine, colorarea grafurilor
       - Schema generală backtracking — înțelege tiparul, nu memoriza

       PROGRAMARE DINAMICĂ (DP):
       - Rucsacul (0/1 knapsack)
       - Cel mai lung subsir crescător (LIS)
       - Numărul minim de monede (coin change)
       - Distanța Levenshtein (edit distance) — opțional avansat
       - REGULA: definește starea, relația de recurență, cazul de bază

       OOP și MEMORIE DINAMICĂ:
       - Python OOP: clase, obiecte, moștenire, polimorfism, __str__, __repr__
       - C++ OOP: clase, constructori, destructori, moștenire
       - Pointeri C++: adresă (&), dereferențiere (*), new, delete
       - Liste dinamice și arbori implementați cu pointeri în C++

       ══════════════════════════════════════════
       CLASA A XII-A — Baze de date, SQL și Machine Learning
       ══════════════════════════════════════════
       BAZE DE DATE RELAȚIONALE:
       - Modelul entitate-relație (ERD):
         → entități, atribute, relații, chei primare (PK) și străine (FK)
         → cardinalități: 1:1, 1:N, N:M (cu entitate de legătură)
         → diagrame ERD pentru scenarii reale (bibliotecă, magazin, școală)
       - Normalizare:
         → FN1 (valori atomice), FN2 (eliminare dependențe parțiale), FN3 (eliminare dependențe tranzitive)
         → dependențe funcționale, descompunerea tabelelor

       SQL — comenzi complete:
       - DDL: CREATE TABLE, ALTER TABLE, DROP TABLE
       - DML: SELECT, INSERT INTO, UPDATE, DELETE
       - Filtrare: WHERE, LIKE, IN, BETWEEN, IS NULL
       - Sortare și grupare: ORDER BY, GROUP BY, HAVING
       - Funcții agregate: COUNT, SUM, AVG, MIN, MAX
       - JOIN-uri: INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN
       - Subinterogări (subqueries)
       - Vizualizări (VIEW): CREATE VIEW
       - Tranzacții: BEGIN, COMMIT, ROLLBACK
       - DCL: GRANT, REVOKE (conceptual)

       PYTHON + BAZE DE DATE:
       - sqlite3: connect, cursor, execute, fetchall, commit
       - mysql.connector — conectare la MySQL (opțional)
       - Executarea SQL din Python, maparea rezultatelor în liste/dicționare

       MACHINE LEARNING cu Python:
       - Pandas: DataFrame, Series, read_csv, head, describe, fillna, groupby
       - NumPy: array, operații vectoriale, dot, reshape, linspace
       - Matplotlib: plot, scatter, bar, hist, xlabel, ylabel, title, show
       - Scikit-learn:
         → train_test_split, fit, predict, score
         → LinearRegression, KNeighborsClassifier, KMeans
         → confusion_matrix, accuracy_score
       - Tipuri de învățare: supervizată (clasificare, regresie) vs. nesupervizată (clustering)
       - Algoritmi introduși: KNN, regresie liniară, K-Means, introducere rețele neuronale
       - PROIECT INTEGRATOR: BD + interfață Python + model ML simplu

       ══════════════════════════════════════════
       REGULI GENERALE INFORMATICĂ:
       ══════════════════════════════════════════
       - COMPLEXITATE: menționează O(n²), O(n log n), O(n) etc. la fiecare algoritm
       - TRACE/URMĂRIRE: arată un exemplu pas cu pas pentru algoritmii importanți
       - ERORI FRECVENTE: semnalează capcanele comune (index out of range, infinit loop, etc.)
       - BAC INFORMATICĂ: examenul folosește C++ sau Pascal — când elevul se pregătește pentru BAC,
         explică și în C++ și menționează că la examen nu se acceptă Python
       - OLIMPIADĂ: problemele de olimpiadă cer de obicei C++ — adaptează explicațiile
""",
    "geografie": r"""
    6. GEOGRAFIE — METODE DIN MANUALUL ROMÂNESC:
       TERMINOLOGIE OBLIGATORIE:
       - Utilizează denumirile oficiale românești: Carpații Meridionali (nu Alpii Transilvani),
         Câmpia Română (nu Câmpia Munteniei), Dunărea (nu Danube)
       - Relief: munte, deal, podiș, câmpie, depresiune, vale, culoar
       - Hidrografie: fluviu, râu, afluent, confluență, debit, regim hidrologic

       PROGRAMA BAC GEOGRAFIE:
       - Geografie fizică: relief, climă, hidrografie, vegetație, soluri, faună
       - Geografie umană: populație, așezări, economie, transporturi
       - Geografie regională: România, Europa, Continente, Probleme globale

       ROMÂNIA — date esențiale de memorat:
       - Suprafață: 238.397 km², Populație: ~19 mil, Capitală: București
       - Cel mai înalt vârf: Moldoveanu (2544m), Cel mai lung râu intern: Mureș
       - Dunărea: intră la Baziaș, iese la Sulina (Delta Dunării — rezervație UNESCO)
       - Regiuni istorice: Transilvania, Muntenia, Moldova, Oltenia, Dobrogea, Banat, Crișana, Maramureș

       DESENE AUTOMATE GEOGRAFIE:
       ✅ Harta schematică România cu regiuni și râuri principale când e cerut
       ✅ Profil de relief (munte-deal-câmpie) ca secțiune transversală
       ✅ Schema circuitului apei în natură
       - Hărți: folosește <path> pentru contururi, NU dreptunghiuri
       - Râuri = linii albastre sinuoase, Munți = triunghiuri sau contururi maro
       - Adaugă ÎNTOTDEAUNA etichete text pentru denumiri
""",
    "istorie": r"""
    7. ISTORIE — METODE DIN MANUALUL ROMÂNESC:
       STRUCTURA OBLIGATORIE pentru orice subiect istoric:
       **Context:** — situația înainte de eveniment
       **Cauze:** — enumerate clar (economice, politice, sociale, externe)
       **Desfășurare:** — cronologie cu date exacte
       **Consecințe:** — pe termen scurt și lung
       **Semnificație istorică:** — de ce contează

       PROGRAMA BAC ISTORIE (CRITIC):
       - Evul Mediu românesc: Întemeierea Țărilor Române (sec. XIV),
         Mircea cel Bătrân, Alexandru cel Bun, Iancu de Hunedoara, Ștefan cel Mare,
         Vlad Țepeș, Mihai Viteazul (prima unire 1600)
       - Epoca modernă: Revoluția de la 1848, Unirea Principatelor 1859 (Cuza),
         Independența 1877-1878, Regatul României, Primul Război Mondial,
         Marea Unire 1918 (1 Decembrie)
       - Epoca contemporană: România interbelică, Al Doilea Război Mondial,
         Comunismul (1947-1989), Revoluția din Decembrie 1989, România post-comunistă
       - Relații internaționale: NATO (2004), UE (2007)

       PERSONALITĂȚI — date exacte:
       - Cuza: domnie 1859-1866, reforme (secularizare, reforma agrară, Codul Civil)
       - Carol I: 1866-1914, Independența 1877, Regatul 1881
       - Ferdinand I: Marea Unire 1918, Regina Maria
       - Nicolae Ceaușescu: 1965-1989, regim totalitar, executat 25 dec. 1989

       ESEUL DE ISTORIE (BAC):
       Structură obligatorie: Introducere (teză) → 2-3 argumente cu surse/date →
       Concluzie. Minim 2 date cronologice și 2 personalități per eseu.
""",
    "limba și literatura română": r"""
    8. LIMBA ȘI LITERATURA ROMÂNĂ — PROGRAMA OFICIALĂ (clasele IX-XII):

       📌 NOTĂ DE CLASĂ: La fiecare răspuns menționează clasa (IX/X/XI/XII) și tipul de
       activitate (analiză text / eseu / gramatică / pregătire BAC).

       NOTAȚII ȘI TERMENI OBLIGATORII:
       - Curent literar: romantism, realism, simbolism, modernism, tradiționism, postmodernism
       - Specii literare: basm cult, nuvelă, roman, poezie lirică, dramă, cronică, eseu
       - Instanțele comunicării: autor, narator, personaj (nu confunda autor cu narator!)
       - Figuri de stil: metaforă, epitet, comparație, personificare, hiperbolă, antiteză,
         enumerație, inversiune, repetiție, anaforă, simbol, alegorie, ironie
       - Prozodie: măsură (silabe), ritm (iamb, troheu, dactil, amfibrah), rimă (împerecheată,
         încrucișată, îmbrățișată, monorimă)
       - NU folosi: „această operă este frumoasă", „autorul vrea să spună"

       ══════════════════════════════════════════
       CLASA A IX-A — Tranziție și baze literare
       ══════════════════════════════════════════

       LITERATURĂ — teme și contexte:
       - De la folclor la literatură cultă: mit, legendă, basm popular → basm cult
       - Umanism, Renaștere, Iluminism în spațiul românesc vs. european
       - Romantism și realism timpuriu (sec. XIX românesc)
       - Identitate individuală/colectivă, istorie națională, cultură populară vs. scrisă

       AUTORI STUDIAȚI (clasa IX):
       Ion Neculce, Anton Pann, Dinicu Golescu, I. Codru-Drăgușanu, Mihail Kogălniceanu,
       Costache Negruzzi, Dimitrie Bolintineanu, Vasile Alecsandri, Grigore Alexandrescu,
       Ion Ghica, Nicolae Filimon, I.L. Caragiale, Ion Creangă, Ioan Slavici
       Autori străini: Machiavelli, Montesquieu, Molière, Lamartine, Stendhal

       LIMBĂ (clasa IX):
       - Evoluția limbii române: origine latină, influențe slave/turcești/grecești/franceze
       - Normă și abatere, dialecte, graiuri, regionalisme, arhaisme, neologisme
       - Construcția textului: coeziune, coerență, topică
       - Comunicare scrisă și orală: e-mail, discurs, dialog argumentativ

       CE PREDĂ PROFESORUL LA IX:
       - Analiză de texte narative și poetice: temă, motiv, personaje, voce narativă
       - Concepte: basm popular/cult, nuvelă, cronică, legendă, realism, romantism
       - Scriere: jurnal de lectură, scurte eseuri de opinie, narațiuni personale
       - Exerciții: ortografie, punctuație, structura frazei, variante regionale

       ══════════════════════════════════════════
       CLASA A X-A — Aprofundare și argumentare
       ══════════════════════════════════════════

       LITERATURĂ — teme și contexte:
       - Romantism și realism românesc (sec. XIX – început XX)
       - Proză: nuvelă și roman realist, roman subiectiv/psihologic
       - Dramaturgie clasică: comedia de moravuri
       - Poezie: de la romantism la simbolism și modernism timpuriu

       OPERE STUDIATE (clasa X):
       - Ion Creangă – Povestea lui Harap-Alb
       - Mihail Sadoveanu – Hanu Ancuței, Baltagul
       - Mircea Eliade – La țigănci, Maitreyi
       - Liviu Rebreanu – Ion
       - Camil Petrescu – Ultima noapte de dragoste, întâia noapte de război
       - Marin Preda – Moromeții
       - I.L. Caragiale – O scrisoare pierdută
       - Mihai Eminescu, Alexandru Macedonski, George Bacovia, Tudor Arghezi,
         Lucian Blaga, Ion Barbu, Nichita Stănescu — poezii reprezentative
       - Ioan Slavici – Moara cu noroc

       LIMBĂ (clasa X):
       - Tipuri de texte: narativ, descriptiv, argumentativ, eseistic
       - Structuri de frază complexe: subordonări, topică marcată
       - Lexic: neologisme, registre stilistice, expresivitate
       - Argumentare scrisă și orală: eseu argumentativ, dezbateri

       CE PREDĂ PROFESORUL LA X:
       - Analize de text pe romane și nuvele: personaje, conflict, perspectivă narativă, temă
       - Compararea a două opere/fragmente (ex: două viziuni asupra satului, două tipuri de erou)
       - Eseuri argumentative pe teme din opere (iubire, război, familie, sat/oraș)
       - Continuarea normelor: greșeli frecvente de ortografie, punctuație, acord gramatical

       ══════════════════════════════════════════
       CLASA A XI-A — Perspectivă istorico-literară
       ══════════════════════════════════════════

       LITERATURĂ — epoci și curente:
       - Umanism și cronicari: Grigore Ureche, Miron Costin, Dimitrie Cantemir
       - Romantismul pașoptist și postpașoptist (Alecsandri, Eminescu etc.)
       - Junimea și Titu Maiorescu (criteriul estetic, direcția nouă)
       - Modernismul interbelic: poezie, proză, teatru

       OPERE STUDIATE (clasa XI):
       - Vasile Alecsandri – Chirița în provincie
       - George Bacovia – poezii; Lucian Blaga – Meșterul Manole
       - Dimitrie Cantemir – Descrierea Moldovei (fragmente)
       - I.L. Caragiale – În vreme de război
       - Miron Costin – Letopisețul Țării Moldovei (fragmente)
       - George Coșbuc – poezii; Octavian Goga – poezii
       - Dacia literară (fragmente programatice — Kogălniceanu)
       - Mircea Eliade – Nuntă în cer
       - Mihai Eminescu – poezii (aprofundat)
       - Ion Neculce – O samă de cuvinte
       - Costache Negruzzi – Alexandru Lăpușneanul
       - Camil Petrescu – Patul lui Procust, Jocul ielelor
       - Liviu Rebreanu – Pădurea spânzuraților, Ciuleandra
       - Ioan Slavici – Moara cu noroc
       - Grigore Ureche – Letopisețul Țării Moldovei (fragmente)

       LIMBĂ (clasa XI):
       - Istoria limbii române: etape cronologice, documente vechi
       - Stilistică: figuri de stil aprofundate, registre de limbă
       - Tipuri de discurs: narativ, descriptiv, argumentativ, expozitiv
       - Pregătire: eseu structurat, rezumat, comentariu literar

       CE PREDĂ PROFESORUL LA XI:
       - Analize literare complexe: relația autor-narator-personaj, simboluri, viziunea autorului
       - Plasarea autorilor pe axa timpului + curentul literar aferent
       - Eseu interpretativ pe text literar — schemă apropiată de subiectele de BAC
       - Prezentarea comparativă a două curente/epoci sau doi autori

       ══════════════════════════════════════════
       CLASA A XII-A — Sinteză și pregătire BAC
       ══════════════════════════════════════════

       LITERATURĂ — recapitulare sistematică:
       Toți autorii canonici: Eminescu, Creangă, Caragiale, Sadoveanu, Slavici, Rebreanu,
       Camil Petrescu, Arghezi, Bacovia, Blaga, Barbu, Nichita Stănescu, Marin Preda,
       G. Călinescu, Marin Sorescu

       OPERE STUDIATE (clasa XII):
       - Tudor Arghezi – poezii (Testament, Flori de mucigai)
       - George Bacovia – poezii (Plumb, Lacustră)
       - Ion Barbu – poezii (Riga Crypto, Joc secund)
       - Lucian Blaga – poezii (Eu nu strivesc corola...)
       - I.L. Caragiale – O scrisoare pierdută
       - George Călinescu – Enigma Otiliei
       - Ion Creangă – Povestea lui Harap-Alb
       - Mihai Eminescu – poezii (Luceafărul, Floare albastră, O, mamă...)
       - Ion Pillat – poezii
       - Marin Preda – Moromeții, Cel mai iubit dintre pământeni
       - Liviu Rebreanu – Ion
       - Mihail Sadoveanu – Baltagul, Hanu Ancuței
       - Ioan Slavici – Moara cu noroc
       - Marin Sorescu – Iona, A treia țeapă
       - Nichita Stănescu – poezii

       ÎNCADRARE CURENTĂ LITERARĂ (STRICT pentru BAC):
       - Romantism: Eminescu — geniu/vulg, natură-oglindă, iubire ideală, timp/spațiu cosmic
       - Simbolism: Bacovia — simboluri, muzicalitate, cromatică depresivă, sinestezii
       - Modernism: Blaga, Arghezi, Barbu, Camil Petrescu — inovație formală, intelectualism
       - Tradiționism: Sadoveanu, Rebreanu (parțial) — specific național, rural, autohtonism
       - Realism: Slavici, Rebreanu, Caragiale — veridicitate, tipologie socială, obiectivitate
       - ⚠️ Creangă (Harap-Alb) = Basm Cult cu specific REALIST (oralitate, umanizarea fantasticului)

       STRUCTURA ESEULUI BAC (OBLIGATORIE):
       Introducere: încadrare autor + operă + curent literar + teză
       Cuprins:
         → Argument 1: idee + citat scurt (max 2 rânduri) + analiză
         → Argument 2: idee + citat + analiză
         → Element de structură/compoziție (titlu, incipit, final, laitmotiv etc.)
         → Limbaj artistic: minim 2 figuri de stil identificate și explicate
       Concluzie: reformularea tezei + judecată de valoare
       - La POEZIE: obligatoriu 1 element de prozodie (măsură, rimă, ritm)
       - La PROZĂ: perspectivă narativă, relație narator-personaj, tehnici narative
       - La DRAMĂ: conflict dramatic, didascalii, limbajul personajelor

       GRAMATICĂ — Subiectul I BAC:
       - Analiză morfologică: parte de vorbire + toate categoriile gramaticale relevante
         → Substantiv: gen, număr, caz, articulare
         → Verb: mod, timp, persoană, număr, diateză
         → Adjectiv: grad de comparație, gen, număr, caz
       - Analiză sintactică: parte de propoziție + funcție sintactică
       - Relații sintactice: coordonare (și, dar, sau, ci, deci, însă) / subordonare (că, să, care, când, dacă)
       - Tipuri de subordonate: subiectivă, predicativă, atributivă, completivă directă/indirectă,
         circumstanțială (de loc, timp, mod, cauză, scop, condiție, concesie)

       TIPURI DE SCRIERE exersate la română (IX-XII):
       - Rezumat: redă obiectiv acțiunea, fără opinii, la persoana a III-a
       - Caracterizare de personaj: trăsături fizice + morale, scene relevante, relații cu alte personaje
       - Comentariu literar: analiză pe text dat, figuri de stil, structură, semnificații
       - Eseu argumentativ: teză + 2 argumente + contraargument (opțional) + concluzie
       - Eseu interpretativ (BAC): schema de mai sus — obligatoriu cu citate și analiză

       CE PREDĂ PROFESORUL LA XII:
       - Recapitulări tematice: autor cu autor, curent cu curent, operă cu operă
       - Simulări de subiecte de BAC cu barem explicit și cronometrare
       - Feedback personalizat pe eseuri scrise de elev
       - Exerciții de gramatică tip Subiectul I (morfologie + sintaxă + vocabular)""",
    "limba engleză": r"""
    9. LIMBA ENGLEZĂ — PROFESOR VIRTUAL (programa MEN, liceu România)

       NIVEL ȘI PROFIL:
       - Clasa IX: consolidare A2→B1 | Clasa X: B1 solid | Clasa XI-XII: B1→B2 (și C1 intensiv)
       - Programa L1 (prima limbă) și L2 (a doua limbă) — abordare diferențiată la cerere
       - Proba C BAC: listening, reading, writing, speaking (niveluri A1–B2)

       TERMINOLOGIE GRAMATICALĂ — întotdeauna în română pentru elevii români:
       - Timp verbal (nu "tense"), Mod (nu "mood"), Voce (activă/pasivă)
       - Propoziție principală / subordonată, Complement direct/indirect

       COMPETENȚE DE EXERSAT (conform programei MEN):
       - Receptare orală: dialoguri, anunțuri, interviuri, prezentări
       - Receptare scrisă: articole, e-mailuri, texte funcționale, texte literare simple
       - Producere orală: prezentări, dezbateri, descrieri, povești
       - Producere scrisă: e-mail, eseu argumentativ, CV, scrisoare de intenție, recenzie
       - Mediere: reformularea/traducerea simplă a unui mesaj din/în română

       TEME PRINCIPALE (programa clasa IX, L1 nouă):
       - Teens' culture, Social media & AI, Books & Movies, Community life
       - Greening life, Hidden tourist gems, Personal growth, Exploring passions

       TEME CLASELE X–XII (4 domenii):
       - Personal: relații, sănătate, timp liber, sport, cultură de tineret
       - Public: societate, economie, mass-media, ecologie, democrație, drepturile omului
       - Ocupațional: profesii, CV, scrisoare de intenție, interviu, etică la locul de muncă
       - Educațional/Cultural: literatură în engleză, civilizație britanică/americană, știință

       GRAMATICĂ — ordinea predării (IX→XII):
       IX:
       - Present/Past Simple & Continuous, Future (will / going to / prez. cont.)
       - Articolele a/an/the/zero article — reguli și excepții
       - Substantiv: singular/plural neregulat, genitiv saxon
       - Pronume: personale, posesive, reflexive
       - Verbe modale de bază: can, must, have to, should
       X:
       - Present Perfect Simple & Continuous (for/since, just/already/yet)
       - Past Perfect; narrative tenses (Simple + Continuous + Perfect)
       - Gerunziu vs. infinitiv (enjoy doing vs. want to do)
       - Vocea pasivă (toate timpurile de bază)
       - Reported speech (concordanța timpurilor, pronume, expresii de timp)
       - Condiționali: tip 1 (real), tip 2 (ireal prezent), tip 3 (ireal trecut), mixt
       XI-XII:
       - Pronume relative (who, which, that, whose) — relative clause definitorii/nedef.
       - Inversiune (Had I known…, Should you need…)
       - Wish / If only (prezent, trecut, viitor)
       - Structuri avansate: despite/in spite of, although/even though, unless, provided that

       STRUCTURA ESEU ARGUMENTATIV (BAC / Cambridge style):
       Introducere (teză clară) → Paragraf 1 (argument + exemplu) →
       Paragraf 2 (argument + exemplu) → Concluzie (reformulare + opinie finală)
       - Topic sentence → development → concluding sentence pentru fiecare paragraf
       - Conectori: Furthermore, However, In addition, On the other hand,
         Despite this, As a result, In conclusion, To sum up

       GREȘELI FRECVENTE (corectează blând, cu explicație):
       - "I am agree" → "I agree" (agree nu e adjectiv)
       - "He go" → "He goes" (prezent simplu, pers. a III-a sg)
       - "more better" → "better" (comparativ neregulat)
       - "I have seen him yesterday" → "I saw him yesterday" (past simple cu moment precis)
       - "She is knowing" → "She knows" (stative verbs nu au continuous)
       - "discuss about" → "discuss" (fără prepoziție)
       - Articol zero la substantive generice: "Life is short" nu "The life is short"

       STIL DE PREDARE:
       - Dă exemple în engleză + traducere/explicație în română
       - La exerciții, arată mai întâi modelul rezolvat, apoi lasă elevul să exerseze
       - Corectura: subliniază greșeala → explică regula → oferă varianta corectă
       - Adaptează nivelul: simplu și concret pentru IX, mai abstract și nuanțat pentru XII
""",
    "limba franceză": r"""
    10. LIMBA FRANCEZĂ — PROFESOR VIRTUAL (programa MEN, liceu România)

        NIVEL ȘI PROFIL:
        - L1 (prima limbă): A2→B1 în clasa IX, țintă B1–B2 la final de liceu
        - L2 (a doua limbă — cea mai frecventă): A2 în IX, B1 în X, B1–B2 în XI-XII
        - Adaptează complexitatea la nivelul cerut de elev

        COMPETENȚE DE EXERSAT (conform programei MEN):
        - Receptare orală: dialoguri simple, anunțuri, mesaje audio/video
        - Receptare scrisă: anunțuri, e-mailuri, postări, articole scurte, broșuri
        - Producere orală: răspunsuri, prezentări scurte, dialoguri, dezbateri simple
        - Producere scrisă: e-mailuri, mesaje, descrieri, narațiuni, texte argumentative
        - Interacțiune și mediere: reformulare, traducere simplă română↔franceză

        TEME PRINCIPALE (4 domenii, programa IX–XII):
        - Personal: familie, prieteni, sănătate, alimentație, timp liber, hobby-uri,
                    universul adolescenței, planuri de viitor
        - Public: orașul/satul, regiuni francofone, călătorii, servicii publice,
                  mass-media, mediu, societate, drepturile omului
        - Ocupațional: meserii, locul de muncă, CV, scrisoare de intenție, interviu,
                       relația angajat-angajator, etică profesională
        - Educațional/Cultural: școala, personalități culturale/științifice/sportive,
                                civilizație franceză și francofonă, literatură simplă

        TEME SPECIALE CLASELE XI–XII:
        - Societate și cetățenie (fake news, globalizare, migrație, diversitate culturală)
        - Cultură și civilizație (scriitori, artiști, evenimente istorice franceze)
        - Texte argumentative complexe (avantaje/dezavantaje, eseu de opinie)

        FUNCȚII DE COMUNICARE — ordinea predării:
        IX: prezentare, salut, cerere/dare de informații, descriere persoane/locuri/obiecte,
            povestire scurtă, cerere/dare permisiune, exprimare acord/dezacord
        X: exprimarea opiniei și susținerea ei, invitație/propunere/acceptare/refuz,
            exprimarea obligației/dorinței/preferinței, indicații de traseu,
            relatare evenimente trecute, formulare ipoteze, exprimarea intenției
        XI-XII: argumentare, prezentare de proiect, dezbatere, mediere/rezumat de text

        GRAMATICĂ — ordinea predării (IX→XII):
        IX:
        - Articol: hotărât (le/la/les), nehotărât (un/une/des), partitiv (du/de la/des)
        - Substantiv + acord adjectiv (gen, număr)
        - Pronume personale subiect, complement (COD/COI), en/y
        - Verbe: présent indicatif (regulate + être, avoir, aller, faire, prendre, venir)
        - Imperativ simplu pentru instrucțiuni
        - Negație: ne...pas, ne...jamais, ne...plus, ne...rien
        - Interogație: Est-ce que…? / inversiune / intonație

        X:
        - Passé composé (avoir/être + participiu trecut) + acord participiu
          → Verbe cu être: aller, venir, partir, arriver, naître, mourir, rester + reflexive
        - Imparfait: formare + utilizare (descrieri, acțiuni repetate în trecut)
        - Passé composé vs. Imparfait — distincție și utilizare împreună în povestire
        - Futur simple: formare + utilizare
        - Condițional prezent: politețe, dorință, ipoteză (Je voudrais…, Si j'avais…)
        - Pronume relative: qui, que, dont, où
        - Adjective: comparativ și superlativ (plus…que, moins…que, le plus…)

        XI-XII:
        - Subjonctif prezent: formare + structuri (il faut que, vouloir que, bien que,
          pour que, à condition que)
        - Condițional trecut: ipoteze ireale despre trecut (Si j'avais su…)
        - Vocea pasivă de bază
        - Propoziții subordonate: cauzale (parce que, puisque), concesive (bien que + subj.),
          condiționale (si + prezent/imperfect/mai-mult-ca-perfect)
        - Discurs indirect (concordanța timpurilor la franceză)

        ACORD PARTICIPIU TRECUT — regulă detaliată (greșeală frecventă!):
        - Cu avoir: acord cu COD plasat ÎNAINTEA verbului
          → "La lettre qu'il a écrite" (COD 'que'=lettre, înainte → acord feminin)
          → "Il a écrit des lettres" (COD după verb → fără acord)
        - Cu être: acord cu subiectul (gen + număr)
          → "Elle est partie", "Ils sont partis"
        - Verbe reflexive → întotdeauna cu être

        STRUCTURA ESEU FRANCEZĂ (BAC / examene):
        Introduction (sujet amené → sujet posé → sujet divisé) →
        Développement: thèse (argument 1 + exemplu) + antithèse (argument 2 + exemplu)
                       + synthèse/nuanță →
        Conclusion (bilan + ouverture)
        - Conectori utili: D'abord, Ensuite, De plus, Cependant, En revanche,
          Néanmoins, Par conséquent, En conclusion, En définitive

        GREȘELI FRECVENTE (corectează blând, cu explicație):
        - Acord adjectiv: "une fille grand" → "une fille grande"
        - Auxiliar greșit: "J'ai allé" → "Je suis allé"
        - Acord participiu cu avoir: "Je l'ai vu" (m.) vs "Je l'ai vue" (f.)
        - Negație incompletă: "Je sais pas" → "Je ne sais pas" (registro formal)
        - Confuzie ser/avoir în expresii: "J'ai faim/froid/chaud" (nu "Je suis faim")
        - Subjonctif omis: "Il faut que tu vas" → "Il faut que tu ailles"

        STIL DE PREDARE:
        - Dă exemple în franceză + traducere/explicație în română
        - Contrastează cu româna sau engleza când ajută înțelegerea
        - La exerciții, arată mai întâi modelul rezolvat, apoi lasă elevul să exerseze
        - Corectura: subliniază greșeala → explică regula → oferă varianta corectă
        - Nivel IX-X: foarte concret, multe exemple, exerciții repetitive
        - Nivel XI-XII: text mai autentic, sarcini de producere liberă, argumentare

""",
    "limba germană": r"""
    11. LIMBA GERMANĂ — PROFESOR VIRTUAL (programa MEN, liceu România)

        NIVEL ȘI PROFIL:
        - L1 (prima limbă): consolidare A1→A2 în IX, țintă B1–B2 la final de liceu
        - L2 (a doua limbă): A2 în IX-X, B1 în XI, B1–B2 în XII
        - L3 (a treia limbă): A1 în IX, A2 în X-XI, B1 în XII
        - Adaptează complexitatea: întreabă elevul clasa și nivelul dacă nu e clar
        - Pregătire BAC: înțelegere text scris, redactare text, probă orală

        COMPETENȚE DE EXERSAT (conform programei MEN + CEFR):
        - Hörverstehen (înțelegere orală): dialoguri, anunțuri, știri scurte audio/video
        - Sprechen (exprimare orală): dialoguri, prezentări, dezbateri ghidate
        - Leseverstehen (înțelegere scrisă): mesaje, e-mailuri, articole, texte funcționale
        - Schreiben (producere scrisă): mesaje, e-mailuri, invitații, CV, eseu scurt
        - Mediation (mediere): rezumarea în română a unui text german și invers,
          traducere funcțională simplă
        - Competență interculturală: comparații România–spațiu germanofon (DE, AT, CH)

        TEME PRINCIPALE — 4 domenii (programa IX–XII):
        - Personal: eu, familia, prietenii, casa, orașul, viața zilnică, sănătate,
                    alimentație, timp liber, hobby-uri, universul adolescenței
        - Public: țări și regiuni germanofone, transport, servicii (magazin, bancă, poștă),
                  mass-media, social media, mediu, societate, cultură europeană
        - Ocupațional: meserii, locul de muncă, CV, scrisoare de intenție, interviu,
                       vocabular profesional (comerț, turism, gastronomie)
        - Educațional/Cultural: viața la liceu, civilizație germanofonă (Germania, Austria,
                                Elveția), personalități, literatură, film, muzică,
                                patrimoniu cultural european

        TEME SPECIALE PE CLASE:
        IX:  Prezentare, familie, școală, casă, oraș, timp liber, sărbători germane
        X:   Rutina zilnică, alimentație, cumpărături, călătorii, media, mediu
        XI:  Relații interpersonale, joburi, CV, teme sociale (migrație, globalizare),
             cultură germanofonă, texte de opinie
        XII: Tehnologie, Europa, Erasmus, texte argumentative, pregătire BAC

        GRAMATICĂ — ordinea predării (IX→XII):
        IX (A1→A2):
        - Genul substantivului (der/die/das) — OBLIGATORIU de memorat cu articol!
          → Sfat: învață întotdeauna substantivul cu articolul (der Tisch, die Lampe, das Buch)
        - Articole hotărâte/nehotărâte la nominativ și acuzativ (ein/eine/ein, kein/keine)
        - Declinarea articolelor la dativ (dem/der/dem, einem/einer/einem)
        - Pronume personale (ich, du, er/sie/es, wir, ihr, sie/Sie)
        - Pronume posesive (mein, dein, sein/ihr, unser, euer, ihr)
        - Verbe la prezent (Präsens): conjugare regulate + neregulate frecvente
          → Neregulate esențiale: sein (bin/bist/ist), haben (habe/hast/hat),
            werden, können, müssen, dürfen, wollen, sollen, mögen, fahren, laufen
        - Ordinea cuvintelor: SVO în propoziția enunțiativă, V la sfârșit în subordonată
        - Propoziție interogativă cu verb la loc 1 (Ist das...?) sau W-Fragen (Was, Wo, Wer...)
        - Numerale, ora, zilele săptămânii, lunile, anotimpurile
        - Imperativ simplu (Geh! Komm! Mach!)
        - Prepoziții frecvente cu acuzativ (durch, für, gegen, ohne, um) și dativ (aus, bei,
          mit, nach, seit, von, zu, gegenüber)

        X (A2→B1):
        - Perfekt (trecut compus): haben/sein + Partizip II
          → Partizip II: ge- + stem + -(e)t (regulate) sau forme neregulate (gehen→gegangen)
          → Verbe cu sein: gehen, kommen, fahren, fliegen, bleiben, sein, werden, sterben
        - Präteritum (trecut narativ): forme frecvente war, hatte, ging, kam, machte
          → În vorbire: Perfekt | În scris/narațiune: Präteritum
        - Comparație adjective: gut → besser → am besten (forme neregulate!)
          → Pozitiv, Komparativ, Superlativ: schnell, schneller, am schnellsten
        - Verbe modale complete: können, müssen, dürfen, wollen, sollen, mögen + Konjunktiv II
        - Conectori: und, aber, oder, denn, weil (V la sfârșit!), dass (V la sfârșit!),
          wenn, obwohl, damit, bevor, nachdem
        - Reflexive Verben (sich waschen, sich freuen, sich interessieren für)

        XI (B1):
        - Subordonate cu weil, dass, wenn, obwohl — VERB MEREU LA SFÂRȘIT
          → "Ich lerne Deutsch, weil es interessant ist." (nu "weil es ist interessant")
        - Pronume relative: der/die/das + declinare (dem, denen, dessen, deren)
          → "Das Buch, das ich lese, ist interessant."
        - Konjunktiv II pentru politețe și ipoteză:
          → ich würde + Infinitiv (würde gehen, würde kaufen)
          → Forme speciale: wäre (sein), hätte (haben), könnte, müsste, dürfte
        - Passiv: werden + Partizip II (Das Auto wird repariert.)
        - Plusquamperfekt (mai-mult-ca-perfect): hatte/war + Partizip II
        - Partizipialkonstruktionen de bază

        XII (B1→B2):
        - Konjunktiv I (vorbire indirectă — Indirekte Rede):
          → Er sagt, er sei krank. / Er sagt, er habe keine Zeit.
        - Structuri avansate: zweiteilige Konnektoren (entweder…oder, sowohl…als auch,
          zwar…aber, nicht nur…sondern auch, weder…noch)
        - Nominalizare și stilul academic/formal
        - Infinitivkonstruktionen cu zu (Es ist wichtig, Deutsch zu lernen.)
        - Genitivul în scris formal (wegen des Wetters, trotz des Regens)

        PARTICULARITĂȚI GERMANE — atenție specială la:
        - GENUL SUBSTANTIVELOR: nu există regulă universală — se memorează cu articol
          → Trucuri utile: -ung, -heit, -keit, -schaft → die (feminin!)
                          -chen, -lein → das (neutru!)
                          -er (agent) → der (de regulă masculin)
        - ORDINEA CUVINTELOR (Satzstellung): verbul conjugat MEREU pe poziția 2
          → "Heute gehe ich in die Schule." (nu "Heute ich gehe...")
        - SEPARABLE VERBEN: prefixul separabil merge la SFÂRȘITUL propoziției
          → "Ich rufe dich an." (anrufen → an...rufe)
        - CAZURILE (Kasus): nominativ, acuzativ, dativ, genitiv — afectează articolul!

        STRUCTURA TEXT ARGUMENTATIV (BAC germană):
        Einleitung (introducere + teză) →
        Hauptteil: Argument 1 + Beispiel → Argument 2 + Beispiel → Gegenargument + Widerlegung →
        Schluss (concluzie + opinie personală)
        - Conectori pentru eseu: Zunächst, Außerdem, Darüber hinaus, Allerdings,
          Dennoch, Obwohl, Deshalb, Zusammenfassend, Meiner Meinung nach

        GREȘELI FRECVENTE (corectează blând, cu explicație):
        - Gen greșit: "der Lampe" → "die Lampe" (atenție la gen!)
        - Verb la poziția greșită: "Heute ich gehe" → "Heute gehe ich" (V pe poz. 2!)
        - Verb la sfârșit omis în subordonată: "weil er ist krank" → "weil er krank ist"
        - Prefix separabil uitat: "Ich rufe dich" → "Ich rufe dich an" (anrufen!)
        - Auxiliar greșit la Perfekt: "Ich habe gegangen" → "Ich bin gegangen"
        - Acuzativ vs dativ confundat: "Ich gehe in dem Park" → "Ich gehe in den Park"
          (mișcare → acuzativ; locație → dativ)
        - Participiu II format greșit: "gegehnt" → "gegangen" (neregulat!)

        STIL DE PREDARE:
        - Explică ÎNTOTDEAUNA genul substantivelor noi: der/die/das + substantiv
        - Contrastează cu română și engleză (ajută mult pentru structura frazei)
        - Ordinea cuvintelor: desenează schema vizual dacă e necesar
          → [Poziția 1] [VERB] [Subiect dacă nu e pe poz.1] [...] [Verb2/Prefix la final]
        - La exerciții: model rezolvat → elev exersează → corecție cu explicație
        - Nivel IX: accent pe vocabular + prezent + câteva verbe neregulate esențiale
        - Nivel X: Perfekt vs Präteritum + modale + conectori
        - Nivel XI-XII: subordonate, Konjunktiv II, texte autentice, producere liberă
""",
}

_PROMPT_ALL_SUBJECTS = "\n    GHID DE COMPORTAMENT:\n" + "".join(_PROMPT_SUBJECTS.values())


def get_system_prompt(materie: str | None = None, pas_cu_pas: bool = False,
                      mod_strategie: bool = False, mod_bac_intensiv: bool = False, mod_avansat: bool = False) -> str:
    """Returnează System Prompt adaptat materiei selectate și modurilor active.
    
    OPTIMIZARE TOKEN: când materia e selectată explicit, include DOAR blocul acelei materii
    (economie 71-94% din tokenii de system prompt față de versiunea completă).
    Când materia e None (Toate materiile), include toate blocurile — comportament original.
    """

    if materie == "pedagogie":
        # Mod pedagogie: trimitem doar _PROMPT_COMUN + _PROMPT_FINAL (fără bloc materie)
        # Economie: ~70-90% din tokenii de system prompt față de versiunea cu materie
        rol_line = (
            "ROL: Ești un profesor și mentor de liceu din România, bărbat, cu experiență în "
            "pregătirea pentru BAC și în strategii de învățare eficientă. "
            "Elevul te întreabă despre cum să învețe mai bine — răspunde ca un mentor experimentat, "
            "concret și personalizat."
        )
    elif materie:
        rol_line = (
            f"ROL: Ești un profesor de liceu din România specializat în {materie.upper()}, "
            f"bărbat, cu experiență în pregătirea pentru BAC. "
            f"Răspunde EXCLUSIV la întrebări legate de {materie}. "
            f"Dacă elevul întreabă despre altă materie, îndrumă-l prietenos să schimbe materia din meniu."
        )
    else:
        rol_line = (
            "ROL: Ești un profesor de liceu din România, universal "
            "(Mate, Fizică, Chimie, Literatură și Gramatică Română, Franceză, Engleză, Germană, "
            "Geografie, Istorie, Informatică, Biologie), bărbat, cu experiență în pregătirea pentru BAC."
        )

    # Bloc suplimentar injectat când modul pas-cu-pas e activ
    pas_cu_pas_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: EXPLICAȚIE PAS CU PAS (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul a activat modul "Pas cu Pas". Respectă OBLIGATORIU aceste reguli pentru ORICE răspuns:

    FORMAT OBLIGATORIU pentru orice problemă sau explicație:
    **📋 Ce avem:**
    - Listează datele cunoscute din problemă

    **🎯 Ce căutăm:**
    - Spune clar ce trebuie aflat/demonstrat

    **🔢 Rezolvare pas cu pas:**
    **Pasul 1 — [nume pas]:** [acțiune + de ce o facem]
    **Pasul 2 — [nume pas]:** [acțiune + de ce o facem]
    ... (continuă până la final)

    **✅ Răspuns final:** [rezultatul clar, cu unități dacă e cazul]

    **💡 Reține:**
    - 1-2 idei cheie de memorat din acest exercițiu

    REGULI STRICTE în modul pas cu pas:
    1. NICIODATĂ nu sări un pas, chiar dacă pare evident.
    2. La fiecare pas explică DE CE faci acea operație, nu doar CE faci.
       - GREȘIT: "Împărțim la 2."
       - CORECT: "Împărțim la 2 pentru că vrem să izolăm variabila x."
    3. Dacă există mai multe metode, alege cea mai simplă și menționeaz-o.
    4. La final, verifică răspunsul (substituie înapoi sau estimează).
    5. Folosește emoji-uri pentru pași (1️⃣, 2️⃣, 3️⃣) dacă sunt mai mult de 3 pași.
    ═══════════════════════════════════════════════════
""" if pas_cu_pas else ""

    # Bloc mod Strategie
    mod_strategie_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: EXPLICĂ-MI STRATEGIA (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul vrea să înțeleagă CUM să gândească rezolvarea, nu să primească calculele gata făcute.

    PENTRU ORICE PROBLEMĂ, răspunde OBLIGATORIU în acest format:

    **🧠 Cum recunoști tipul de problemă:**
    - Ce elemente din enunț îți spun că e acest tip de exercițiu
    - Cu ce tip de problemă să nu o confunzi

    **🗺️ Strategia de rezolvare (fără calcule):**
    - Pasul 1: Ce faci primul și DE CE
    - Pasul 2: Unde vrei să ajungi
    - Pasul 3: Ce formulă/metodă folosești și de ce pe aceasta și nu alta

    **⚠️ Capcane frecvente:**
    - Greșelile tipice pe care le fac elevii la acest tip de problemă

    **✏️ Acum încearcă tu:**
    - Ghidează elevul să aplice strategia, nu îi da răspunsul direct

    REGULI STRICTE:
    1. NU calcula nimic — explică doar logica și gândirea
    2. Dacă elevul are lipsuri de teorie pentru a rezolva, explică ÎNTÂI teoria necesară
    3. Folosește analogii și exemple din viața reală pentru a face strategia memorabilă
    ═══════════════════════════════════════════════════
""" if mod_strategie else ""

    # Bloc mod BAC Intensiv
    mod_bac_intensiv_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: PREGĂTIRE BAC INTENSIVĂ (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul este în clasa a 12-a și se pregătește intens pentru BAC. Adaptează TOATE răspunsurile:

    PRIORITIZARE CONȚINUT:
    1. Focusează-te EXCLUSIV pe ce apare la BAC — nu preda lucruri care nu sunt în programă
    2. La fiecare răspuns, menționează: "Acesta apare frecvent la BAC" sau "Rar la BAC, dar posibil"
    3. Când explici o metodă, precizează dacă e metoda acceptată la BAC sau există variante mai scurte

    FORMAT RĂSPUNS BAC:
    - Structurează exact ca la subiectele de BAC (Subiectul I / II / III)
    - Punctaj estimativ: "Acest tip de problemă valorează ~15 puncte la BAC"
    - Timp estimativ: "La BAC ai ~8 minute pentru acest tip"

    TEORIA LIPSĂ — DETECTARE AUTOMATĂ (CRITIC):
    Dacă observi că elevul nu are baza teoretică pentru a rezolva problema:
    1. OPREȘTE-TE din rezolvare
    2. Spune explicit: "⚠️ Înainte să rezolvăm, trebuie să știi teoria din spate:"
    3. Explică teoria necesară SCURT și CLAR (definiție + formulă + exemplu simplu)
    4. Abia apoi continuă cu rezolvarea problemei originale

    SFATURI BAC specifice:
    - Reamintește elevului să verifice răspunsul când mai are timp
    - Semnalează când o problemă are "capcane" tipice de BAC
    - La Română: reamintește structura eseului și punctajul pe competențe
    ═══════════════════════════════════════════════════
""" if mod_bac_intensiv else r"""

    TEORIA LIPSĂ — DETECTARE AUTOMATĂ:
    Dacă observi că elevul nu are baza teoretică pentru a rezolva problema:
    1. OPREȘTE-TE și spune: "⚠️ Pentru asta trebuie să știi mai întâi:"
    2. Explică teoria necesară pe scurt (definiție + formulă + exemplu)
    3. Apoi continuă cu rezolvarea
"""

    mod_avansat_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: AVANSAT (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul știe deja bazele și NU vrea explicații de la zero.

    REGULI STRICTE în Mod Avansat:
    1. NU explica concepte de bază — presupune că le știe
    2. Mergi DIRECT la ideea cheie, metoda sau formula relevantă
    3. Răspuns scurt și dens: maxim 3-5 rânduri pentru o problemă tipică
    4. Format preferat:
       💡 **Ideea:** [ce metodă/formulă se aplică și de ce]
       ⚡ **Calcul rapid:** [doar pașii esențiali, fără explicații evidente]
       ✅ **Rezultat:** [răspunsul final]
    5. Dacă elevul greșește abordarea, corectează DIRECT: "Nu, aplică X în loc de Y."
    6. Folosește notații scurte și simboluri matematice, nu propoziții lungi
    ═══════════════════════════════════════════════════
""" if mod_avansat else ""

    # ── Selectează blocul de materie ──
    if materie == "pedagogie":
        # Mod pedagogie: fără bloc de materie — _PROMPT_COMUN conține deja tot ce trebuie
        ghid_materie = ""
    elif materie and materie in _PROMPT_SUBJECTS:
        # OPTIMIZARE: doar blocul materiei selectate
        ghid_materie = "\n    GHID DE COMPORTAMENT:\n" + _PROMPT_SUBJECTS[materie]
    else:
        # Toate materiile (sau materie necunoscută) — comportament original
        ghid_materie = _PROMPT_ALL_SUBJECTS

    return ("ROL: " + rol_line
            + pas_cu_pas_bloc
            + mod_strategie_bloc
            + mod_bac_intensiv_bloc
            + mod_avansat_bloc
            + _PROMPT_COMUN
            + ghid_materie
            + _PROMPT_FINAL)



# System prompt inițial — ține cont de modul pas cu pas dacă era deja setat
SYSTEM_PROMPT = get_system_prompt(
    materie=None,
    pas_cu_pas=st.session_state.get("pas_cu_pas", False),
    mod_avansat=st.session_state.get("mod_avansat", False),
    mod_strategie=st.session_state.get("mod_strategie", False),
    mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
)


# === DETECȚIE AUTOMATĂ MATERIE ===
# Mapare cuvinte cheie → materie (pentru detecție rapidă fără apel API)
SUBJECT_KEYWORDS = {
    "matematică": [
        "ecuație", "ecuatia", "funcție", "functie", "derivată", "derivata", "integrală", "integrala",
        "limită", "limita", "matrice", "determinant", "trigonometrie", "geometrie", "algebră", "algebra",
        "logaritm", "radical", "inecuație", "inecuatia", "probabilitate", "combinatorică",
        "vector", "plan", "dreapta", "paralelă", "perpendiculară", "triunghi", "cerc", "parabola",
        "matematica", "mate", "math", "calcul", "număr", "numărul", "numere",
    ],
    "fizică": [
        "forță", "forta", "viteză", "viteza", "accelerație", "acceleratie", "masă", "masa",
        "energie", "putere", "curent electric", "tensiune electrică", "rezistență electrică",
        "curent", "tensiune", "rezistenta", "rezistență", "circuit", "circuit electric",
        "circuit serie", "circuit paralel", "serie", "paralel",
        "câmp", "camp", "undă", "unda", "optică", "optica", "lentilă", "lentila",
        "termodinamică", "termodinamica", "gaz", "presiune", "volum", "temperatură", "temperatura",
        "fizica", "fizică", "mecanică", "mecanica", "electricitate", "baterie", "condensator",
        "gravitație", "gravitatie", "frecare", "pendul", "oscilatie", "oscilație",
        "rezistor", "ohm", "amper", "volt", "watt", "joule", "newton",
        "nod", "ramură", "legea lui kirchhoff", "legea lui ohm",
    ],
    "chimie": [
        "atom", "moleculă", "molecula", "element chimic", "compus chimic",
        "reacție chimică", "reactie chimica", "ecuație chimică",
        "acid", "sare", "oxidare", "reducere", "electroliză", "electroliza",
        "număr de moli", "masă molară", "stoechiometrie",
        "organic", "alcan", "alchenă", "alchena", "alcool", "ester", "chimica", "chimie",
        "ph", "soluție", "solutie", "concentratie", "concentrație",
        "hidrogen", "oxigen", "carbon", "azot", "legătură chimică",
    ],
    "biologie": [
        "celulă", "celula", "adn", "arn", "proteină", "proteina", "enzimă", "enzima",
        "mitoză", "mitoza", "meioză", "meioza", "genetică", "genetica", "cromozom",
        "fotosinteza", "fotosinteză", "respiratie", "respirație", "metabolism",
        "ecosistem", "specie", "organ", "tesut", "țesut", "sistem nervos",
        "biologie", "biologic", "planta", "plantă", "animal",
    ],
    "informatică": [
        # general
        "algoritm", "cod", "program", "informatica", "informatică", "programare",
        # Python keywords
        "python", "def ", "list", "dict", "tuple", "set(", "append", "pandas", "numpy",
        "matplotlib", "scikit", "sklearn", "dataframe", "tkinter", "sqlite", "flask",
        # C++ keywords
        "c++", "cout", "cin", "#include", "vector<", "struct ", "pointer", "new ",
        # structuri de date
        "functie", "funcție", "vector", "array", "stivă", "stiva", "coada", "coadă",
        "lista inlantuita", "listă înlănțuită", "arbore", "graf", "heap",
        # algoritmi
        "backtracking", "greedy", "recursivitate", "recursiv", "sortare", "cautare",
        "bubble sort", "merge sort", "quicksort", "dijkstra", "bfs", "dfs",
        "programare dinamica", "programare dinamică", "rucsac", "backtrack",
        "complexitate", "recursie",
        # BD si SQL
        "sql", "baza de date", "bază de date", "select ", "join", "create table",
        "entitate", "normalizare", "sqlite", "mysql",
        # ML
        "machine learning", "invatare automata", "învățare automată", "knn",
        "clustering", "kmeans", "regresie", "clasificare", "neural", "scikit",
        # pseudocod
        "pseudocod", "variabila", "variabilă", "ciclu", "for ", "while ", "if ",
    ],
    "geografie": [
        "relief", "munte", "câmpie", "campie", "râu", "rau", "dunărea", "dunarea",
        "climă", "clima", "vegetatie", "vegetație", "populație", "populatie",
        "romania", "românia", "europa", "continent", "ocean", "geografie",
        "carpati", "carpații", "câmpia", "campia", "delta", "lac",
    ],
    "istorie": [
        "război", "razboi", "revoluție", "revolutie", "unire", "independenta", "independență",
        "cuza", "eminescu", "mihai viteazul", "stefan cel mare", "ștefan cel mare",
        "comunism", "comunist", "ceausescu", "ceaușescu", "bac 1918", "marea unire",
        "medieval", "evul mediu", "modern", "contemporan", "istorie", "istoric",
        "domnie", "domitor", "rege", "regat", "principat",
    ],
    "limba și literatura română": [
        "roman", "roman", "poezie", "poem", "eminescu", "rebreanu", "sadoveanu",
        "preda", "arghezi", "blaga", "bacovia", "caragiale", "creanga", "creangă",
        "eseu", "comentariu", "caracterizare", "narator", "personaj", "tema",
        "figuri de stil", "metafora", "metaforă", "epitet", "comparatie", "comparație",
        "roman", "proza", "proză", "dramaturgie", "gramatica", "gramatică",
        "romana", "română", "literatura", "literatură",
    ],
    "limba engleză": [
        # Identificatori de limbă / materie
        "english", "engleză", "engleza", "grammar", "essay", "vocabulary",
        # Structuri gramaticale exclusiv engleze (fraze compuse — fără risc de false positive)
        "present perfect", "past simple", "past tense", "future tense",
        "present tense", "conditional tense", "passive voice", "reported speech",
        "modal verb", "relative clause", "indirect speech",
        # Teme din programa IX (L1 nouă)
        "teens culture", "social media", "influencer", "personal growth",
        "community life", "tourist gems", "greening life",
        # Teme X–XII (4 domenii)
        "cover letter", "job interview", "curriculum vitae",
        "british culture", "american culture", "civilizație britanică",
        # Tipuri de texte / sarcini frecvente
        "formal letter", "informal email", "book review", "film review",
        "argumentative essay", "opinion essay", "for and against",
        # Vocabular gramatical în română, specific englezei
        "gerunziu", "infinitiv", "vocea pasivă", "inversiune",
        "propoziție relativă", "vorbire indirectă", "condiționala de tip",
    ],
    "limba franceză": [
        # Identificatori de limbă / materie
        "français", "franceză", "franceza",
        # Timpuri verbale exclusiv franceze
        "passé composé", "imparfait", "subjonctif", "futur simple",
        "conditionnel", "participe passé", "plus-que-parfait",
        # Verbe auxiliare (formă exclusiv franceză)
        "être", "avoir",
        # Articole și structuri exclusiv franceze
        "article partitif", "article défini", "article indéfini",
        "du ", "de la ", "des ",
        # Teme din programă
        "civilizație franceză", "francofonă", "francofonie", "espace francophone",
        "pays francophones",
        # Funcții de comunicare frecvente în lecții
        "accord du participe", "accord adjectif", "auxiliaire être",
        "auxiliaire avoir", "verbe pronominal", "verbe réfléchi",
        # Vocabular gramatical în română, specific francezei
        "participiu trecut", "acord participiu", "verb reflexiv",
        "propoziție relativă franceză", "subjonctiv francez",
    ],
    "limba germană": [
        # Identificatori de limbă / materie
        "germană", "germana", "deutsch", "německy", "allemand",
        # Terminologie exclusiv germană
        "der ", "die ", "das ", "ein ", "eine ", "kein", "keine",
        "umlaut", "eszett", "ß",
        # Timpuri verbale exclusiv germane
        "perfekt", "präteritum", "plusquamperfekt", "konjunktiv",
        "konjunktiv ii", "futur i", "futur ii",
        # Structuri gramaticale exclusiv germane
        "separable verben", "trennbare verben", "reflexive verben",
        "verb la sfârșitul", "verb la sfarsitul", "satzstellung",
        "partizip ii", "partizip i",
        # Cazuri germane
        "nominativ", "acuzativ", "dativ", "genitiv",
        # Verbe modale germane
        "können", "müssen", "dürfen", "wollen", "sollen", "mögen",
        # Vocabular gramatical în română specific germanei
        "genul substantivului", "articol hotărât german", "declinare germană",
        "propoziție subordonată germană", "prefix separabil",
        # Teme culturale specifice
        "spațiu germanofon", "germanofon", "hörverstehen", "leseverstehen",
        "oktoberfest", "bundesrepublik",
    ],
}


# Cuvinte care sunt exclusive unei materii — boost mare dacă apar
_STRONG_INDICATORS = {
    # IMPORTANT: folosiți doar cuvinte complete sau fraze — NU substring-uri scurte
    # care pot apărea accidental în alte cuvinte (ex: "ion" e în "funcționează").
    "informatică":  ["python", "c++", "def ", "cout", "#include", "algoritm", "recursiv",
                     "backtracking", "pandas", "sklearn", "compilator", "pseudocod"],
    "matematică":   ["ecuație", "inecuație", "derivată", "integrală", "matrice", "determinant",
                     "funcție", "progresie", "logaritm", "trigonometrie"],
    "fizică":       ["forță", "viteză", "accelerație", "curent electric", "tensiune electrică",
                     "rezistență electrică", "câmp magnetic", "undă", "frecvență",
                     "energie cinetică", "circuit electric", "circuit serie", "circuit paralel",
                     "lege lui ohm", "legea lui ohm", "condensator", "inductor",
                     "câmp electric", "sarcină electrică", "putere electrică"],
    "chimie":       ["reacție chimică", "ecuație chimică", "mol ", "moli ", "masă molară",
                     "oxidare", "reducere", "electroliză", "hidroliza",
                     "acid tare", "bază tare", "soluție tampon", "concentrație molară",
                     "legătură covalentă", "legătură ionică", "orbital"],
    "biologie":     ["celulă", "adn", "arn", "proteină", "metabolism", "fotosinteză",
                     "ecosistem", "evoluție", "genetică", "cromozom", "mitoză"],
    "istorie":      ["război mondial", "tratat de pace", "revoluție", "regat", "imperiu",
                     "dinastie", "domnie", "bătălie"],
    "geografie":    ["relief", "climă", "populație", "hidrografie", "câmpie", "munte",
                     "râu", "bazin hidrografic"],
    "limba și literatura română": ["figuri de stil", "narator", "personaj principal",
                     "comentariu literar", "caracterizare", "metaforă", "epitet",
                     "curent literar", "roman realist"],
    # Indicatori puternici pentru limbi străine — fraze exclusiv din terminologia
    # gramaticală a limbii respective, imposibil de confundat cu româna sau altă materie
    "limba engleză": [
        # Timpuri verbale în engleză (formă exclusiv engleză)
        "present perfect", "past simple", "past tense", "future tense",
        "present continuous", "past continuous", "past perfect",
        # Structuri gramaticale exclusiv engleze
        "passive voice", "reported speech", "modal verb", "relative clause",
        "conditional sentence", "indirect speech", "gerund", "infinitive",
        # Tipuri de texte / sarcini BAC engleză
        "argumentative essay", "opinion essay", "formal letter", "book review",
        "for and against essay",
        # Teme specifice programei noi clasa IX
        "teens culture", "greening life", "personal growth",
    ],
    "limba franceză": [
        # Timpuri verbale în franceză (formă exclusiv franceză)
        "passé composé", "imparfait", "subjonctif", "futur simple",
        "conditionnel présent", "conditionnel passé", "plus-que-parfait",
        # Structuri gramaticale exclusiv franceze
        "participe passé", "être ou avoir", "accord du participe",
        "article partitif", "verbe pronominal", "pronom relatif",
        # Teme specifice programei franceze
        "espace francophone", "pays francophones", "civilisation française",
        # Conectori/structuri de eseu francez
        "thèse antithèse", "plan dialectique",
    ],
    "limba germană": [
        # Timpuri verbale exclusiv germane
        "perfekt", "präteritum", "plusquamperfekt",
        "konjunktiv ii", "konjunktiv i",
        # Structuri gramaticale exclusiv germane — imposibil de confundat
        "separable verben", "trennbare verben", "partizip ii",
        "verb la sfârșitul propoziției", "satzstellung",
        # Cazuri germane (formă exclusiv germană)
        "der den dem des", "akkusativ", "nominativ kasus",
        # Verbe auxiliare în contexte germane
        "haben oder sein", "sein oder haben",
        # Conectori cu verb la sfârșit (exclusiv germani)
        "weil verb", "obwohl verb", "damit verb",
        # Teme culturale specifice germanei
        "spațiu germanofon", "germanofonă", "hörverstehen", "leseverstehen",
        "bundesrepublik", "österreich deutsch",
    ],
}

def detect_subject_from_text(text: str) -> str | None:
    """Detectează materia dintr-un text folosind cuvinte cheie cu sistem de ponderi.
    
    Folosește indicatori puternici (boost x3) + indicatori generali + penalizări încrucișate.
    Evită false positive-uri de tip 'matrice' → matematică când e informatică.
    """
    text_lower = text.lower()
    scores = {}

    # Scor de bază din cuvintele cheie generale
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[subject] = score

    # Boost x3 pentru indicatori puternici (specific unui singur domeniu)
    for subject, indicators in _STRONG_INDICATORS.items():
        strong_hits = sum(1 for ind in indicators if ind in text_lower)
        scores[subject] = scores.get(subject, 0) + strong_hits * 3

    # Penalizare încrucișată: dacă avem indicatori puternici de informatică,
    # penalizăm matematica (ex: "matrice" în context cod → nu matematică)
    info_strong = sum(1 for ind in _STRONG_INDICATORS["informatică"] if ind in text_lower)
    if info_strong >= 2:
        scores["matematică"] = scores.get("matematică", 0) * 0.3

    # Elimină scoruri 0 și returnează maximul cu threshold minim
    scores = {s: v for s, v in scores.items() if v > 0}
    if not scores:
        return None
    best = max(scores, key=scores.get)
    # Trebuie să fie clar câștigător — dacă e egal cu al doilea, nu detectăm
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] == sorted_scores[1]:
        return None
    return best


def get_detected_subject() -> str | None:
    """Returnează materia detectată din session_state sau None."""
    return st.session_state.get("_detected_subject", None)


def update_system_prompt_for_subject(materie: str | None):
    """Actualizează system prompt-ul pentru materia dată și salvează în session_state.
    Resetează și flag-ul de caching — noul prompt trebuie re-cached la primul apel.
    """
    st.session_state["_detected_subject"] = materie
    # Invalidăm caching-ul local — promptul s-a schimbat, cache-ul vechi nu mai e valid
    st.session_state["_ctx_cache_enabled"] = True   # permite re-caching cu noul prompt
    global _prompt_cache_store
    _prompt_cache_store = {}  # curăță toate intrările locale
    st.session_state["system_prompt"] = get_system_prompt(
        materie=materie,
        pas_cu_pas=st.session_state.get("pas_cu_pas", False),
        mod_avansat=st.session_state.get("mod_avansat", False),
        mod_strategie=st.session_state.get("mod_strategie", False),
        mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
    )




safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]



# ============================================================
# === SIMULARE BAC ===
# ============================================================

MATERII_BAC = {
    "📐 Matematică M1": {
        "cod": "matematica_m1",
        "profile": ["M1 - Mate-Info"],
        "subiecte": ["Numere complexe", "Funcții", "Ecuații/inecuații", "Probabilități", "Geometrie analitică", "Matrice și sisteme", "Legi de compoziție", "Derivate și monotonie", "Integrale și limite"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": True,
        "structura": {
            "S1": "6 exerciții scurte × 5p = 30p",
            "S2": "2 probleme (matrice+sisteme, lege compoziție) = 30p",
            "S3": "2 probleme (funcții+derivate, integrale/limite) = 30p",
        },
    },
    "⚡ Fizică tehnologic": {
        "cod": "fizica_tehnologic",
        "profile": ["Filiera tehnologică"],
        "subiecte": ["Mecanică", "Termodinamică", "Curent continuu", "Optică"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": True,
        "structura": {
            "arii": "4 arii (A-Mecanică, B-Termodinamică, C-Curent continuu, D-Optică)",
            "alegere": "Candidatul alege 2 arii din 4",
            "per_arie": "S.I (5 grilă × 3p) + S.II (problemă 15p) + S.III (problemă 15p)",
        },
    },
    "📖 Română real/tehn": {
        "cod": "romana_real_tehn",
        "profile": ["Real/tehnologic"],
        "subiecte": ["Text la prima vedere", "Comentariu literar", "Eseu personaj/curent"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": True,
        "structura": {
            "S1": "50p: A (5 itemi 30p) + B (text argumentativ 150+ cuvinte, 20p)",
            "S2": "10p: comentariu 50+ cuvinte pe fragment literar",
            "S3": "30p: eseu 400+ cuvinte (personaj/text narativ/curent literar)",
        },
    },
    "📐 Matematică M2": {
        "cod": "matematica_m2",
        "profile": ["M2 - Științe ale naturii"],
        "subiecte": ["Funcții", "Ecuații/inecuații", "Probabilități", "Geometrie", "Derivate", "Integrale"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": False,
    },
    "🧪 Chimie": {
        "cod": "chimie",
        "profile": ["Chimie anorganică", "Chimie organică"],
        "subiecte": ["Chimie anorganică", "Chimie organică"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": False,
    },
    "🧬 Biologie": {
        "cod": "biologie",
        "profile": ["Biologie vegetală și animală", "Anatomie și fiziologie umană"],
        "subiecte": ["Anatomie", "Genetică", "Ecologie"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": False,
    },
    "🏛️ Istorie": {
        "cod": "istorie",
        "profile": ["Umanist", "Pedagogic"],
        "subiecte": ["Istorie românească", "Istorie universală"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": False,
    },
    "🌍 Geografie": {
        "cod": "geografie",
        "profile": ["Profiluri umaniste"],
        "subiecte": ["Geografia României", "Geografia Europei", "Demografie"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": False,
    },
    "💻 Informatică": {
        "cod": "informatica",
        "profile": ["C++", "Pascal"],
        "subiecte": ["Algoritmi", "Structuri de date", "Programare completă"],
        "timp_minute": 180,
        "punctaj_total": 100,
        "date_reale": False,
    },
}

# ── Date reale BAC 2021-2025 ────────────────────────────────────────────────
BAC_DATE_REALE = {
    "matematica_m1": {
        "tipare": [
            "Numere complexe: calcul cu z, modul, argument, verificare egalități",
            "Funcții simple f(f(x)), f(x+a), f(f(m))=valoare — verificare proprietăți",
            "Ecuații exponențiale (3ˣ, 2ˣ) sau logaritmice (log₃, log)",
            "Probabilități cu numere naturale de două cifre (cifra zecilor, cifra unităților, multipli)",
            "Geometrie analitică: drepte perpendiculare/paralele, coordonate punct, distanțe",
            "Trigonometrie: triunghi dreptunghic/isoscel, arie, sin2A, cos A, tgB",
            "Matrice 3×3 cu parametru: det(A(a)), inversabilitate, proprietăți det(A·B)",
            "Lege de compoziție x★y: calcule punctuale, element neutru, inegalități, condiții",
            "Funcție cu ln sau eˣ: derivată, monotonie, extreme, ecuație f(x)=0 soluție unică",
            "Integrală definită + limită: calcul ∫, primitive, lim(1/x)∫₀ˣtf(t)dt",
        ],
        "subiecte_reale": [
            {"an": 2021, "s1": "Media aritmetică a=b=2021/2 | f(x)=2x²-3x+1, A(1,m) pe grafic | log₃(x+3)-log₃(x+2)=2 | Mulțime 16 submulțimi | M(3,0)N(8,3)P(6,3): MN⃗+MP⃗=MQ⃗ | sin2A=cosA·sinA → A=π/4", "s2": "A(a) 3×3 cu log a: det=1, inversabilă, det(A(a)·A(a+1)⁻¹)≥8 | x★y=xy+m(x+y)+m², m>0: calcule, 2★1=5→2★5=1, (3-x)★(3-x)=m", "s3": "f(x)=4x²-2x-4lnx: f'=(4x²-4x-1)/x, monotonie, exact 2 soluții f(x)=0 | f(x)=(4x²+1)/(x²+1): ∫₀¹f=11/3, asimptotă oblică, ∫₀¹G(x)dx=π/3+ln2-4/3"},
            {"an": 2022, "s1": "(8-6√6)(6√6+1)=2 | f(x)=x+3m, f(f(m))=2m | 2³ˣ·2²=4·4ˣ | P(cifra zecilor | divizor 6) | y=3x-2, A(a,a) pe dreaptă | Triunghi isoscel AB=10, cosA=0, arie=50", "s2": "A(x) 3×3: det=1, A(x)·A(y)=A(x+y), A(n)²+A(n)³+2A(n)=O | x★y=x²y²-4(x+y)²+1: 1★0=-3, e=0 neutru, x★x=4", "s3": "f(x)=x-ln(x²+x+5): f'=(x²-9x)/(x²+x+5), monotonie, f(x)=m soluție unică | f(x)=x³-3x+9: ∫₅⁹f=0, ∫₀⁴x dx=∫₀⁴(f-x)dx, limₙ Iₙ=0"},
            {"an": 2023, "s1": "z=3+i: (z²-zi)=10 | f(x)=x+5: f(x)²-f(x²)=1 | x³-3x²+2x=2 | P(5n+5 multiplu 10) | A(4,0)B(5,4): dreaptă prin origine paralelă AB | Triunghi isoscel dreptunghic în A, arie=4 → BC=4", "s2": "A(a) 3×3 + sistem: det(A(0))=8, inversabilitate, a=-2: x₀+z₀=2 | x★y=x²+y²-2x²y²: 2★3=18, e=1 neutru, x★(1-x)≤1", "s3": "f(x)=(3lnx+1)/(x-1): f'=(x²+15)/(x-1)², asimptotă oblică, (3lnx+1)/(x-1)≥1 | f(x)=x²+2x+eˣ: integrale, lim(1/x)∫₀ˣtf(t)dt=1"},
            {"an": 2024, "s1": "Progresie aritmetică a₂=14, a₃=18 → a₁=? | f(x)=x+2: f(f(5))=9 | 3ˣ+2ˣ⁺³=2ˣ⁻¹ | Numere impare 2 cifre din {1,2,3,7,9} | A(2,1): 2AB=OA | Triunghi dreptunghic BC=12, BC/AB=2 → arie=18√3", "s2": "A și B(x): det(B)=1, B(x)B(y)=B(x+y)-xyA, B(x)B(1-x)=A | f(X)=X³+2X²+X+a-2: f(1)=4, rădăcini a=2, (x₁-1)(x₂-2)(x₃-3)=4", "s3": "f(x)=x²-2x+2eˣ: f', lim f'/f, imaginea f | f(x)=(4x²+2)/(6x+1): ∫₁²f=12/11"},
            {"an": 2025, "s1": "z₁=1-i, z₂=2+i: 2z₁+iz₂=1 | f(x)=x+3: f(f(a))=9 | 2x²-3x-2=0 | P(număr 2 cifre, divizor 6²) | A(0,1)B(5,0)C(6,3)D(a,b): AC și BD același mijloc | Triunghi dreptunghic AB=2, tgB=√3 → BC=2√10", "s2": "A(x) 3×3: det(A(-1))=8, A(x)A(y)=A(x+y), A(x)²+A(x)³+2A(x)=O | f(X)=X³-3X²-6X+a: f(1)=-3, câit+rest la g=X²+X-3, (x₁+1)(x₂+1)(x₃+1)=1", "s3": "f(x)=x²+lnx+2: f'=2x+1/x, asimptotă oblică, bijectivă | f(x)=(x²+3)/(x+1): ∫₀³f=30, ∫₀¹xf=1-ln2, arie cu g=f(x)/eˣ egală cu (1/2)(e-1)/(e+1)"},
        ],
    },
    "fizica_tehnolog": {
        "tipare": {
            "mecanica": ["Mișcare rectilinie — energie, viteză, forțe pe corp", "Unități de măsură SI pentru mărimi derivate (putere, lucru mecanic, energie)", "Plan înclinat — forță de frecare, unghi, coeficient μ", "Sistem corpuri legate prin fir — tensiune, accelerație, mase", "Corp pe plan înclinat cu forță de tracțiune — lucru mecanic, energie cinetică, viteză"],
            "termodinamica": ["Transformări termodinamice — proprietăți izotermă/izobară/izochoră/adiabatică", "ΔU, căldură schimbată, lucru mecanic — formule și calcule", "Gaz ideal în cilindru cu piston — presiune, volum, temperatură, densitate", "Ciclu termodinamic p-V sau p-T — energie internă, căldură, lucru mecanic total"],
            "curent": ["Putere maximă transferată consumatorului (R=r)", "Rezistența unui conductor — ρ, lungime, secțiune", "Circuit serie-paralel cu sursă — tensiuni, intensități, rezistențe echivalente", "Două consumatoare în paralel — intensitate, energie, putere disipată"],
            "optica": ["Refracție și reflexie — relații unghiuri, indice de refracție n·sin·i=sin·r", "Lentilă convergentă — mărire, distanțe, focală, construcție imagine", "Efect fotoelectric — energia fotonului, frecvența prag, energia cinetică", "Lamă cu fețe plane paralele — drum optic, unghi refracție, viteza luminii"],
        },
    },
    "romana_real_tehn": {
        "tipare_s1_itemi": [
            "1. Indică sensul din text al cuvântului X și al secvenței Y",
            "2. Menționează o caracteristică/profesie/statut al personajului X, valorificând textul",
            "3. Precizează momentul/reacția/trăsătura morală + justifică cu o secvență din text",
            "4. Explică motivul pentru care... / reprezintă un eveniment / are loc situația X",
            "5. Prezintă în 30-50 cuvinte atmosfera/atitudinea/o situație conform textului",
        ],
        "teme_argumentativ": [
            "importanța studiului / lecturii / educației",
            "influența profesorilor / mentorilor asupra elevilor",
            "rolul culturii în formarea personalității",
            "comportamentul social / responsabilitatea individuală",
            "influența înfățișării/imaginii asupra succesului personal",
            "importanța relațiilor umane / familiei / prieteniei",
        ],
        "tipare_s2": [
            "Prezintă, în minimum 50 de cuvinte, perspectiva narativă din fragmentul de mai jos.",
            "Prezintă, în minimum 50 de cuvinte, rolul notațiilor autorului în fragmentul de mai jos.",
            "Comentează, în minimum 50 de cuvinte, relația dintre ideea poetică și mijloacele artistice în textul dat.",
        ],
        "repere_s3": [
            "1. Prezentarea statutului social, psihologic, moral al personajului ales",
            "2. Evidențierea unei trăsături prin două episoade sau secvențe comentate",
            "3. Analiza a două elemente de structură/compoziție/limbaj (acțiune, conflict, tehnici narative, modalități de caracterizare, registre stilistice)",
        ],
        "autori_opere": [
            "Ion Creangă — Ion / Harap-Alb / Amintiri din copilărie",
            "Ioan Slavici — Moara cu noroc / Mara / Popa Tanda",
            "Liviu Rebreanu — Ion / Pădurea spânzuraților",
            "Camil Petrescu — Ultima noapte de dragoste / Patul lui Procust",
            "G. Călinescu — Enigma Otiliei",
            "G.M. Zamfirescu — Domnișoara Nastasia / Maidanul cu dragoste",
            "Mihail Sadoveanu — Baltagul / Frații Jderi",
        ],
        "subiecte_reale": [
            {"an": 2022, "s1_text": "Text despre critici literari (Basil Munteanu & Vladimir Streinu)", "s1_B": "text argumentativ 150-200 cuvinte (succes, cultură etc.)", "s2": "Prezentarea rolului notațiilor autorului în fragment dramatic (50+ cuvinte)", "s3": "Eseu personaj dintr-un basm cult (ex. Harap-Alb): statut + trăsătură prin 2 episoade + 2 elemente structură/limbaj"},
            {"an": 2023, "s1_text": "Text la prima vedere — 5 întrebări standard", "s1_B": "text argumentativ 150+ cuvinte", "s2": "Comentariu relație idee poetică — mijloace artistice (50+ cuvinte)", "s3": "Eseu 400+ cuvinte personaj dintr-o nuvelă/roman din literatura română"},
            {"an": 2024, "s1_text": "Fragment memorialistic despre Sadoveanu", "s1_B": "text argumentativ despre cultură/comportament social (150+ cuvinte)", "s2": "Prezintă în min. 50 cuvinte perspectiva narativă din fragmentul dat", "s3": "Eseu 400+ cuvinte personaj dintr-un text dramatic/narativ studiat"},
            {"an": 2025, "s1_text": "Fragment despre profesorul Vasile Pârvan (Grigore Băjenaru, 'Părintele Geticei')", "s1_itemi": "1.sensul 'prielnic'+'pe timpuri' | 2.caracteristică profesori cu săli pline | 3.momentul cursului Pârvan+secvență | 4.motivul referinței la originea numelui | 5.atmosfera sălii Odobescu în 30-50 cuvinte", "s1_B": "Argumentează dacă înfățișarea poate influența succesul, cu referire la text și experiență personală/culturală (150+ cuvinte)", "s2": "Rolul notațiilor autorului în fragmentul din 'Domnișoara Nastasia' de G.M. Zamfirescu — scena cu Vulpașin și Nastasia (50+ cuvinte)", "s3": "Eseu min. 400 cuvinte: particularitățile de construcție ale unui personaj dintr-un text narativ studiat de Ion Creangă sau Ioan Slavici. Repere: statut social/psihologic/moral; trăsătură prin 2 episoade; 2 elemente structură/compoziție/limbaj"},
        ],
    },
}




def extract_text_from_photo(image_bytes: bytes, materie_label: str) -> str:
    """Extrage textul scris de mână dintr-o fotografie folosind Groq Vision (base64)."""
    import base64
    try:
        key = keys[st.session_state.get("key_index", 0)]
        client = GroqClient(api_key=key)

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            f"Ești un asistent care transcrie text scris de mână din lucrări de elevi la {materie_label}. "
            f"Transcrie EXACT tot ce este scris în imagine, inclusiv formule, simboluri matematice și calcule. "
            f"Păstrează structura (Subiectul I, II, III dacă există). "
            f"Dacă un cuvânt e greu de citit, transcrie-l cu [?]. "
            f"Nu adăuga nimic, nu corecta nimic — transcrie fidel."
        )
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[Eroare la citirea pozei: {e}]"


def get_bac_prompt_ai(materie_label, materie_info, profil):
    cod = materie_info.get("cod", "")
    date_reale = materie_info.get("date_reale", False)  # FIX bug 10: folosit în fallback generic

    # ── MATEMATICĂ M1 — date reale 2021-2025 ──
    if cod == "matematica_m1":
        data = BAC_DATE_REALE["matematica_m1"]
        tipare = data["tipare"]
        # Alege un subiect real ca referință (random)
        ref = random.choice(data["subiecte_reale"])
        tipare_str = "\n".join(f"  - {t}" for t in tipare)
        return (
            f"Generează un subiect COMPLET de BAC la Matematică M1 (mate-info), "
            f"IDENTIC ca structură și dificultate cu subiectele oficiale române din 2021-2025.\n\n"
            f"STRUCTURĂ EXACTĂ (obligatorie):\n"
            f"SUBIECTUL I (30 de puncte) — 6 exerciții × 5p fiecare:\n"
            f"  Tipare care se repetă an de an:\n{tipare_str}\n\n"
            f"SUBIECTUL al II-lea (30 de puncte) — 2 probleme structurate (a, b, c):\n"
            f"  Problema 1: matrice 3×3 cu parametru real — det, inversabilitate, proprietăți\n"
            f"  Problema 2: lege de compoziție pe ℝ — calcule punctuale, element neutru, inegalități\n\n"
            f"SUBIECTUL al III-lea (30 de puncte) — 2 probleme structurate (a, b, c):\n"
            f"  Problema 1: funcție cu ln sau eˣ — arătați f'(x), monotonie, soluție unică f(x)=0\n"
            f"  Problema 2: integrală definită — calculați ∫, proprietate integrală, limită tip lim(1/x)∫₀ˣ\n\n"
            f"REFERINȚĂ (subiect real {ref['an']}):\n"
            f"  S.I: {ref['s1']}\n"
            f"  S.II: {ref['s2']}\n"
            f"  S.III: {ref['s3']}\n\n"
            f"IMPORTANT:\n"
            f"- Folosește numere și funcții DIFERITE față de exemplul de referință\n"
            f"- Dificultatea trebuie să fie realistă pentru BAC național\n"
            f"- Formulează cerințele exact ca la examen ('Arătați că...', 'Determinați...', 'Demonstrați că...')\n"
            f"- 10 puncte din oficiu\n\n"
            f"La final adaugă baremul:\n"
            f"[[BAREM_BAC]]\n"
            f"SUBIECTUL I: [răspunsurile corecte pentru fiecare item]\n"
            f"SUBIECTUL al II-lea: [soluțiile complete pas cu pas]\n"
            f"SUBIECTUL al III-lea: [soluțiile complete pas cu pas]\n"
            f"[[/BAREM_BAC]]"
        )

    # ── FIZICĂ TEHNOLOGIC — date reale 2021-2025 ──
    elif cod == "fizica_tehnolog":
        data = BAC_DATE_REALE["fizica_tehnolog"]
        tipare = data["tipare"]
        # Alege 2 arii random pentru subiect
        arii_disponibile = ["A. MECANICĂ", "B. TERMODINAMICĂ", "C. CURENT CONTINUU", "D. OPTICĂ"]
        arii_alese = random.sample(arii_disponibile, 2)
        tipare_mec = "\n".join(f"    - {t}" for t in tipare["mecanica"])
        tipare_term = "\n".join(f"    - {t}" for t in tipare["termodinamica"])
        tipare_cur = "\n".join(f"    - {t}" for t in tipare["curent"])
        tipare_opt = "\n".join(f"    - {t}" for t in tipare["optica"])
        return (
            f"Generează un subiect COMPLET de BAC la Fizică — filiera tehnologică, "
            f"IDENTIC ca structură cu subiectele oficiale române din 2021-2025.\n\n"
            f"STRUCTURĂ EXACTĂ:\n"
            f"Subiectul are 4 ARII tematice (A–D). Candidatul rezolvă DOAR 2 la alegere.\n"
            f"Generează TOATE cele 4 arii. Pentru fiecare arie:\n"
            f"  - Subiectul I (15 puncte): 5 itemi tip GRILĂ (a, b, c, d) × 3p\n"
            f"  - Subiectul II (15 puncte): o problemă structurată cu 4 cerințe (a, b, c, d)\n"
            f"  - Subiectul III (15 puncte): o problemă mai complexă cu 4 cerințe (a, b, c, d)\n\n"
            f"TIPARE REALE PE ARII:\n"
            f"A. MECANICĂ:\n{tipare_mec}\n\n"
            f"B. TERMODINAMICĂ:\n{tipare_term}\n\n"
            f"C. CURENT CONTINUU:\n{tipare_cur}\n\n"
            f"D. OPTICĂ:\n{tipare_opt}\n\n"
            f"IMPORTANT:\n"
            f"- Datele numerice trebuie să fie realiste și să dea calcule curate\n"
            f"- Formulează grilele cu exact 4 variante, dintre care exact una corectă\n"
            f"- Problemele din S.II și S.III trebuie să fie rezolvabile pas cu pas\n"
            f"- Indică la fiecare arie: 'Aria A — Mecanică', etc.\n"
            f"- 10 puncte din oficiu\n\n"
            f"[[BAREM_BAC]]\n"
            f"ARIA A: [răspunsuri grilă + soluții probleme]\n"
            f"ARIA B: [răspunsuri grilă + soluții probleme]\n"
            f"ARIA C: [răspunsuri grilă + soluții probleme]\n"
            f"ARIA D: [răspunsuri grilă + soluții probleme]\n"
            f"[[/BAREM_BAC]]"
        )

    # ── ROMÂNĂ REAL/TEHNOLOGIC — date reale 2021-2025 ──
    elif cod == "romana_real_tehn":
        data = BAC_DATE_REALE["romana_real_tehn"]
        ref = random.choice(data["subiecte_reale"])
        itemi_str = "\n".join(f"  {it}" for it in data["tipare_s1_itemi"])
        teme_str = "\n".join(f"  - {t}" for t in data["teme_argumentativ"])
        s2_str = "\n".join(f"  - {t}" for t in data["tipare_s2"])
        repere_str = "\n".join(f"  {r}" for r in data["repere_s3"])
        autori_str = "\n".join(f"  - {a}" for a in data["autori_opere"])
        return (
            f"Generează un subiect COMPLET de BAC la Limba și literatura română — profil real/tehnologic, "
            f"IDENTIC ca structură cu subiectele oficiale din 2021-2025.\n\n"
            f"STRUCTURĂ EXACTĂ:\n\n"
            f"SUBIECTUL I (50 de puncte):\n"
            f"Partea A (30 puncte) — Text la prima vedere (proză, memorialistică sau publicistică, 1-2 pagini).\n"
            f"Generează un text original de 200-300 cuvinte, apoi formulează EXACT 5 cerințe:\n"
            f"{itemi_str}\n\n"
            f"Partea B (20 puncte) — Text argumentativ de minimum 150 cuvinte pe o temă din text:\n"
            f"  Alege una dintre temele frecvente:\n{teme_str}\n"
            f"  Cerința standard: 'Redactează un text de minimum 150 de cuvinte, în care să argumentezi dacă [tema], "
            f"raportându-te atât la informațiile din textul dat, cât și la experiența personală sau culturală.'\n\n"
            f"SUBIECTUL al II-lea (10 puncte):\n"
            f"  Un fragment literar scurt (dramatic sau liric) + una din cerințele:\n{s2_str}\n\n"
            f"SUBIECTUL al III-lea (30 de puncte):\n"
            f"  Eseu de minimum 400 de cuvinte. Alege un autor și operă din:\n{autori_str}\n"
            f"  Formularea standard: 'Redactează un eseu de minimum 400 de cuvinte, în care să prezinți "
            f"particularitățile de construcție ale unui personaj dintr-un text narativ studiat.'\n"
            f"  Repere obligatorii (în barem):\n{repere_str}\n\n"
            f"REFERINȚĂ (structura subiectului real {ref['an']}):\n"
            f"  S.I text: {ref.get('s1_text', 'text la prima vedere')}\n"
            f"  S.II: {ref.get('s2', '')}\n"
            f"  S.III: {ref.get('s3', '')}\n\n"
            f"IMPORTANT:\n"
            f"- Textul de la S.I trebuie să fie original, coerent, de nivel liceal\n"
            f"- Fragmentul de la S.II trebuie să fie dintr-o operă reală din programa de liceu\n"
            f"- 10 puncte din oficiu\n\n"
            f"[[BAREM_BAC]]\n"
            f"SUBIECTUL I — Partea A: [răspunsurile așteptate pentru fiecare cerință + punctaj]\n"
            f"SUBIECTUL I — Partea B: [criterii text argumentativ + punctaj]\n"
            f"SUBIECTUL al II-lea: [răspuns așteptat + criterii + punctaj]\n"
            f"SUBIECTUL al III-lea: [repere eseu + criterii conținut (18p) + redactare (12p)]\n"
            f"[[/BAREM_BAC]]"
        )

    # ── ALTE MATERII — prompt generic îmbunătățit ──
    else:
        # FIX bug 4: .get() cu fallbackuri sigure — structura dicționarului poate varia
        subiecte = materie_info.get("subiecte", [])
        subiecte_str = ", ".join(subiecte) if subiecte else materie_label
        structura = materie_info.get("structura", {})
        structura_str = "\n".join(f"  {k}: {v}" for k, v in structura.items()) if structura else ""
        timp = materie_info.get("timp_minute", 180)
        # FIX bug 10: date_reale folosit — hint explicit către AI pentru calitate
        sursa_hint = (
            "Inspiră-te din tipare reale ale subiectelor BAC din 2021–2025 pentru această materie.\n"
            if date_reale else
            "Generează un subiect realist, la nivel BAC, respectând programa românească.\n"
        )
        return (
            f"Generează un subiect complet de BAC la {materie_label} ({profil}), "
            f"identic ca structură și dificultate cu subiectele oficiale din România.\n\n"
            f"{sursa_hint}"
            f"STRUCTURĂ OBLIGATORIE:\n"
            f"- SUBIECTUL I (30 puncte): itemi obiectivi/semiobiectivi\n"
            f"- SUBIECTUL al II-lea (30 puncte): probleme/analiză structurată\n"
            f"- SUBIECTUL al III-lea (30 puncte): problemă complexă / eseu / sinteză\n"
            f"- 10 puncte din oficiu\n\n"
            f"TEME: {subiecte_str}\n"
            f"TIMP: {timp} minute\n\n"
            f"La final adaugă baremul:\n"
            f"[[BAREM_BAC]]\n"
            f"SUBIECTUL I: [răspunsuri și punctaj]\n"
            f"SUBIECTUL al II-lea: [soluții și punctaj]\n"
            f"SUBIECTUL al III-lea: [criterii și punctaj]\n"
            f"[[/BAREM_BAC]]"
        )


def get_bac_correction_prompt(materie_label, subiect, raspuns_elev, from_photo=False):
    source_note = (
        "NOTĂ: Răspunsul a fost extras automat dintr-o fotografie a lucrării. "
        "Unele cuvinte pot fi transcrise imperfect din cauza scrisului de mână — "
        "judecă după intenția elevului, nu după eventuale erori de OCR.\n\n"
        if from_photo else ""
    )

    # Reguli de limbaj adaptate materiei
    if "Română" in materie_label:
        lang_rules = (
            "CORECTARE LIMBĂ ROMÂNĂ (OBLIGATORIU — punctaj separat):\n"
            "- Ortografie și punctuație (virgule, punct, ghilimele «»)\n"
            "- Acordul gramatical (subiect-predicat, adjectiv-substantiv)\n"
            "- Folosirea corectă a cratimei, apostrofului\n"
            "- Exprimare clară, coerentă, fără pleonasme sau cacofonii\n"
            "- Registru stilistic adecvat eseului de BAC\n"
            "- Acordă până la 10 puncte bonus/penalizare pentru calitatea limbii\n\n"
        )
    else:
        lang_rules = (
            f"CORECTARE LIMBAJ ȘTIINȚIFIC ({materie_label}):\n"
            "- Terminologie specifică folosită corect\n"
            "- Notații și simboluri respectate (ex: m pentru masă, nu M; v nu V pentru viteză)\n"
            "- Unități de măsură scrise corect și complet\n"
            "- Formulele scrise corect, fără ambiguități\n"
            "- Raționament logic și coerent exprimat în cuvinte\n"
            "- Acordă până la 5 puncte bonus/penalizare pentru calitatea exprimării\n\n"
        )

    return (
        f"Ești examinator BAC România pentru {materie_label}.\n\n"
        f"{source_note}"
        f"SUBIECTUL:\n{subiect}\n\n"
        f"RĂSPUNSUL ELEVULUI:\n{raspuns_elev}\n\n"
        f"Corectează COMPLET în această ordine:\n\n"
        f"## 📊 Punctaj per subiect\n"
        f"- Subiectul I: X/30 puncte\n"
        f"- Subiectul II: X/30 puncte\n"
        f"- Subiectul III: X/30 puncte\n"
        f"- Din oficiu: 10 puncte\n\n"
        f"## ✅ Ce a făcut bine\n"
        f"[aspecte corecte]\n\n"
        f"## ❌ Greșeli și explicații\n"
        f"[fiecare greșeală explicată]\n\n"
        f"## 🖊️ Calitatea limbii și exprimării\n"
        f"{lang_rules}"
        f"## 🎓 Nota finală\n"
        f"**Nota: X/10** — [verdict scurt]\n\n"
        f"## 💡 Recomandări pentru BAC\n"
        f"[2-3 sfaturi concrete]\n\n"
        f"Fii constructiv, cald, dar riguros ca un examinator real."
    )


def parse_bac_subject(response):
    """Parsează răspunsul AI în subiect + barem.
    FIX bug 16: dacă AI-ul nu generează baremul în tags, căutăm secțiunea 'BAREM' în text."""
    barem = ""
    subject_text = response
    match = re.search(r"\[\[BAREM_BAC\]\](.*?)\[\[/BAREM_BAC\]\]", response, re.DOTALL)
    if match:
        barem = match.group(1).strip()
        subject_text = response[:match.start()].strip()
    else:
        # FIX bug 16: fallback — caută o secțiune de barem neîncadrată în tags
        # AI-ul uneori scrie "BAREM:" sau "## Barem" fără tag-uri
        barem_match = re.search(
            r'\n(?:##\s*)?(?:BAREM|Barem|barem)[:\s]+(.*)',
            response, re.DOTALL | re.IGNORECASE
        )
        if barem_match:
            barem = barem_match.group(1).strip()
            subject_text = response[:barem_match.start()].strip()
        # Dacă tot nu găsim barem, subject_text rămâne tot textul (comportament original)
    return subject_text, barem


def format_timer(seconds_remaining):
    h = seconds_remaining // 3600
    m = (seconds_remaining % 3600) // 60
    s = seconds_remaining % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_bac_sim_ui():
    st.subheader("🎓 Simulare BAC")

    # ── ECRAN DE START ──
    if not st.session_state.get("bac_active"):
        col1, col2 = st.columns(2)
        with col1:
            bac_materie = st.selectbox("📚 Materia:", options=list(MATERII_BAC.keys()), key="bac_mat_sel")
            info = MATERII_BAC[bac_materie]
            bac_profil = st.selectbox("🎯 Profil:", options=info["profile"], key="bac_prof_sel")
        with col2:
            bac_tip = "🤖 Generat de AI"
            use_timer = st.checkbox(f"⏱️ Cronometru ({info['timp_minute']} min)", value=True, key="bac_timer")

        # Info card — diferit pt materii cu date reale vs fără
        if info.get("date_reale"):
            structura = info.get("structura", {})
            structura_html = "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in structura.items())
            st.markdown(
                "<div style='background:linear-gradient(135deg,#11998e,#38ef7d);"
                "color:white;padding:18px 22px;border-radius:12px;margin:12px 0'>"
                "<h4 style='margin:0 0 8px 0'>✅ Subiecte bazate pe tipare reale BAC 2021–2025</h4>"
                f"<ul style='margin:0;padding-left:18px;line-height:1.9'>{structura_html}</ul>"
                "<p style='margin:10px 0 0 0;font-size:13px;opacity:0.9'>"
                "⏱️ 3 ore · 100 puncte (90p scrise + 10p oficiu)</p>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background:linear-gradient(135deg,#667eea,#764ba2);"
                "color:white;padding:18px 22px;border-radius:12px;margin:12px 0'>"
                "<h4 style='margin:0 0 8px 0'>📋 Subiect generat de AI</h4>"
                "<ul style='margin:0;padding-left:18px;line-height:1.8'>"
                "<li>Structură inspirată din modelele BAC oficiale</li>"
                "<li>Rezolvi în timp real cu cronometru opțional</li>"
                "<li>Primești corectare AI detaliată + barem</li>"
                "</ul></div>",
                unsafe_allow_html=True
            )

        st.divider()
        col_s, col_b = st.columns(2)
        with col_s:
            btn_lbl = "🚀 Generează subiect AI"
            if st.button(btn_lbl, type="primary", use_container_width=True):
                with st.spinner("📝 Se generează subiectul BAC..."):
                    prompt = get_bac_prompt_ai(bac_materie, info, bac_profil)
                    full = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(
                            materie=None,
                            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                            mod_avansat=st.session_state.get("mod_avansat", False),
                            mod_strategie=st.session_state.get("mod_strategie", False),
                            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                        )
                    ))
                subject_text, barem = parse_bac_subject(full)


                st.session_state.update({
                    "bac_active": True,
                    "bac_materie": bac_materie,
                    "bac_profil": bac_profil,
                    "bac_tip": bac_tip,
                    "bac_subject": subject_text,
                    "bac_barem": barem,
                    "bac_raspuns": "",
                    "bac_corectat": False,
                    "bac_corectare": "",
                    "bac_start_time": time.time() if use_timer else None,
                    "bac_timp_min": info["timp_minute"],
                    "bac_use_timer": use_timer,
                })
                st.rerun()
        with col_b:
            if st.button("↩️ Înapoi la chat", use_container_width=True):
                st.session_state.pop("bac_mode", None)
                st.rerun()
        return

    # ── SIMULARE ACTIVĂ ──
    col_title, col_timer = st.columns([3, 1])
    with col_title:
        st.markdown(f"### {st.session_state.bac_materie} · {st.session_state.bac_profil}")
    with col_timer:
        if st.session_state.get("bac_use_timer") and st.session_state.get("bac_start_time"):
            elapsed = int(time.time() - st.session_state.bac_start_time)
            total   = st.session_state.bac_timp_min * 60
            left    = max(0, total - elapsed)
            pct     = left / total
            color   = "#2ecc71" if pct > 0.5 else ("#e67e22" if pct > 0.2 else "#e74c3c")
            st.markdown(
                f'<div style="background:{color};color:white;padding:8px 12px;'
                f'border-radius:8px;text-align:center;font-size:20px;font-weight:700">'
                f'⏱️ {format_timer(left)}</div>',
                unsafe_allow_html=True
            )
            if left == 0:
                st.warning("⏰ Timpul a expirat!")
                # FIX bug 15: la expirarea timpului, trimite automat răspunsul curent
                # dacă elevul nu a trimis deja și există un răspuns scris
                if (
                    not st.session_state.get("bac_corectat")
                    and not st.session_state.get("bac_timer_submitted")
                    and st.session_state.get("bac_raspuns", "").strip()
                ):
                    st.session_state["bac_timer_submitted"] = True
                    with st.spinner("⏰ Timp expirat — se corectează automat..."):
                        _prompt_timeout = get_bac_correction_prompt(
                            st.session_state.bac_materie,
                            st.session_state.bac_subject,
                            st.session_state.bac_raspuns,
                            from_photo=st.session_state.get("bac_from_photo", False),
                        )
                        _corectare_timeout = "".join(run_chat_with_rotation(
                            [], [_prompt_timeout],
                            system_prompt=get_system_prompt(
                                materie=MATERII.get(st.session_state.bac_materie),
                                pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                                mod_avansat=st.session_state.get("mod_avansat", False),
                                mod_strategie=st.session_state.get("mod_strategie", False),
                                mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                            )
                        ))
                    st.session_state.bac_corectare = _corectare_timeout
                    st.session_state.bac_corectat  = True
                    st.rerun()
            elif left > 0 and not st.session_state.get("bac_corectat"):
                # Nu bloca serverul cu sleep — folosim JS pentru countdown
                components.html(
                    "<script>setTimeout(() => window.parent.location.reload(), 1000);</script>",
                    height=0
                )
                st.stop()

    st.divider()

    with st.expander("📋 Subiectul", expanded=not st.session_state.bac_corectat):
        st.markdown(st.session_state.bac_subject)

    if not st.session_state.bac_corectat:
        st.markdown("### ✏️ Răspunsurile tale")

        tab_foto, tab_text = st.tabs(["📷 Fotografiază lucrarea", "⌨️ Scrie manual"])

        raspuns = st.session_state.get("bac_raspuns", "")
        from_photo = False

        # ── TAB FOTO ──
        with tab_foto:
            st.info(
                "📱 **Pe telefon:** apasă butonul de mai jos și fotografiază lucrarea.\n\n"
                "💻 **Pe calculator:** încarcă o poză din galerie.\n\n"
                "AI-ul va citi textul și va porni corectarea automat."
            )
            uploaded_photo = st.file_uploader(
                "Încarcă fotografia lucrării:",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="bac_photo_upload",
                help="Fă o poză clară, cu lumină bună, la lucrarea scrisă de mână."
            )

            if uploaded_photo:
                st.image(uploaded_photo, caption="Fotografia încărcată", use_container_width=True)

                if not st.session_state.get("bac_ocr_done"):
                    with st.spinner("🔍 Profesorul citește lucrarea..."):
                        img_bytes = uploaded_photo.read()
                        text_extras = extract_text_from_photo(img_bytes, st.session_state.bac_materie)
                    st.session_state.bac_raspuns  = text_extras
                    st.session_state.bac_ocr_done = True
                    st.session_state.bac_from_photo = True

                    # Pornește corectura automat
                    with st.spinner("📊 Se corectează lucrarea..."):
                        prompt = get_bac_correction_prompt(
                            st.session_state.bac_materie,
                            st.session_state.bac_subject,
                            text_extras,
                            from_photo=True
                        )
                        corectare = "".join(run_chat_with_rotation(
                            [], [prompt],
                            system_prompt=get_system_prompt(
                                materie=MATERII.get(st.session_state.bac_materie),
                                pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                                mod_avansat=st.session_state.get("mod_avansat", False),
                                mod_strategie=st.session_state.get("mod_strategie", False),
                                mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                            )
                        ))
                    st.session_state.bac_corectare = corectare
                    st.session_state.bac_corectat  = True
                    st.rerun()

                if st.session_state.get("bac_ocr_done"):
                    with st.expander("📄 Text extras din poză", expanded=False):
                        st.text(st.session_state.get("bac_raspuns", ""))

        # ── TAB TEXT ──
        with tab_text:
            raspuns = st.text_area(
                "Scrie rezolvarea completă:",
                value=st.session_state.get("bac_raspuns", ""),
                height=350,
                placeholder="Subiectul I:\n1. ...\n2. ...\n\nSubiectul II:\n...\n\nSubiectul III:\n...",
                key="bac_ans_input"
            )
            st.session_state.bac_raspuns = raspuns
            st.session_state.bac_from_photo = False

            if st.button("🤖 Corectare AI", type="primary", use_container_width=True,
                         disabled=not raspuns.strip()):
                with st.spinner("📊 Se corectează lucrarea..."):
                    prompt = get_bac_correction_prompt(
                        st.session_state.bac_materie,
                        st.session_state.bac_subject,
                        raspuns,
                        from_photo=False
                    )
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(
                            materie=MATERII.get(st.session_state.bac_materie),
                            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                            mod_avansat=st.session_state.get("mod_avansat", False),
                            mod_strategie=st.session_state.get("mod_strategie", False),
                            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                        )
                    ))
                st.session_state.bac_corectare = corectare
                st.session_state.bac_corectat  = True
                st.rerun()

        st.divider()
        col_barem, col_nou = st.columns(2)
        with col_barem:
            if st.session_state.get("bac_barem"):
                if st.button("📋 Arată Baremul", use_container_width=True):
                    st.session_state.bac_show_barem = not st.session_state.get("bac_show_barem", False)
                    st.rerun()
        with col_nou:
            if st.button("🔄 Subiect nou", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.rerun()

        if st.session_state.get("bac_show_barem") and st.session_state.get("bac_barem"):
            with st.expander("📋 Barem de corectare", expanded=True):
                st.markdown(st.session_state.bac_barem)

    else:
        st.markdown("### 📊 Corectare AI")
        st.markdown(st.session_state.bac_corectare)
        if st.session_state.get("bac_barem"):
            with st.expander("📋 Barem"):
                st.markdown(st.session_state.bac_barem)
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Subiect nou", type="primary", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("✏️ Reîncerc același subiect", use_container_width=True):
                st.session_state.bac_corectat  = False
                st.session_state.bac_corectare = ""
                st.session_state.bac_raspuns   = ""
                if st.session_state.get("bac_use_timer"):
                    st.session_state.bac_start_time = time.time()
                st.rerun()
        with col3:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.session_state.pop("bac_mode", None)
                st.rerun()


# ============================================================
# === CORECTARE TEME ===
# ============================================================

def get_homework_correction_prompt(materie_label: str, text_tema: str, from_photo: bool = False) -> str:
    source_note = (
        "NOTĂ: Tema a fost extrasă dintr-o fotografie. "
        "Unele cuvinte pot fi transcrise imperfect — judecă după intenția elevului.\n\n"
        if from_photo else ""
    )

    if "Română" in materie_label:
        corectare_limba = (
            "## 🖊️ Corectare limbă și stil\n"
            "Acordă atenție specială:\n"
            "- **Ortografie**: diacritice (ă,â,î,ș,ț), cratimă, apostrof\n"
            "- **Punctuație**: virgulă, punct, linie de dialog, ghilimele «»\n"
            "- **Acord gramatical**: subiect-predicat, adjectiv-substantiv, pronume\n"
            "- **Exprimare**: cacofonii, pleonasme, tautologii, registru stilistic\n"
            "- **Coerență**: logica textului, legătura dintre idei\n"
            "Subliniază greșelile găsite și explică regula corectă.\n\n"
        )
    else:
        corectare_limba = (
            f"## 🖊️ Limbaj și exprimare ({materie_label})\n"
            "- Terminologie specifică folosită corect\n"
            "- Notații, simboluri și unități de măsură corecte\n"
            "- Raționament exprimat clar și logic\n\n"
        )

    return (
        f"Ești profesor de {materie_label} și corectezi tema unui elev de liceu.\n\n"
        f"{source_note}"
        f"TEMA ELEVULUI:\n{text_tema}\n\n"
        f"Corectează complet și constructiv:\n\n"
        f"## ✅ Ce a făcut bine\n"
        f"[aspecte corecte — fii specific, nu generic]\n\n"
        f"## ❌ Greșeli de conținut\n"
        f"[fiecare greșeală de materie explicată, cu varianta corectă]\n\n"
        f"{corectare_limba}"
        f"## 📊 Notă orientativă\n"
        f"**Nota: X/10** — [justificare scurtă]\n\n"
        f"## 💡 Sfaturi pentru data viitoare\n"
        f"[2-3 recomandări concrete și aplicabile]\n\n"
        f"Ton: cald, constructiv, ca un profesor care vrea să ajute, nu să descurajeze."
    )


def run_homework_ui():
    st.subheader("📚 Corectare Temă")

    if not st.session_state.get("hw_done"):
        col1, col2 = st.columns([2, 1])
        with col1:
            hw_materie = st.selectbox(
                "📚 Materia temei:",
                options=[m for m in MATERII.keys() if m != "🤖 Automat"],
                key="hw_materie_sel"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Profesorul se adaptează materiei.")

        st.divider()

        tab_foto, tab_text = st.tabs(["📷 Fotografiază tema", "⌨️ Scrie / lipește textul"])

        with tab_foto:
            st.info(
                "📱 **Pe telefon:** fotografiază caietul sau foaia de temă.\n\n"
                "💻 **Pe calculator:** încarcă o poză din galerie.\n\n"
                "Profesorul va citi și corecta automat."
            )
            hw_photo = st.file_uploader(
                "Încarcă fotografia temei:",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="hw_photo_upload",
                help="Asigură-te că poza e clară și bine luminată."
            )

            if hw_photo and not st.session_state.get("hw_ocr_done"):
                st.image(hw_photo, caption="Fotografia încărcată", use_container_width=True)
                with st.spinner("🔍 Profesorul citește tema..."):
                    text_extras = extract_text_from_photo(hw_photo.read(), hw_materie)
                st.session_state.hw_text       = text_extras
                st.session_state.hw_ocr_done   = True
                st.session_state.hw_from_photo = True
                st.session_state.hw_materie    = hw_materie
                with st.spinner("📝 Se corectează tema..."):
                    prompt = get_homework_correction_prompt(hw_materie, text_extras, from_photo=True)
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(
                            materie=MATERII.get(hw_materie),
                            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                            mod_avansat=st.session_state.get("mod_avansat", False),
                            mod_strategie=st.session_state.get("mod_strategie", False),
                            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                        )
                    ))
                st.session_state.hw_corectare = corectare
                st.session_state.hw_done      = True
                st.rerun()
            elif hw_photo and st.session_state.get("hw_ocr_done"):
                with st.expander("📄 Text extras din poză", expanded=False):
                    st.text(st.session_state.get("hw_text", ""))

        with tab_text:
            hw_text = st.text_area(
                "Lipește sau scrie textul temei:",
                value=st.session_state.get("hw_text", ""),
                height=300,
                placeholder="Scrie sau lipește tema aici...",
                key="hw_text_input"
            )
            st.session_state.hw_text = hw_text
            if st.button("📝 Corectează tema", type="primary",
                         use_container_width=True, disabled=not hw_text.strip()):
                st.session_state.hw_materie    = hw_materie
                st.session_state.hw_from_photo = False
                with st.spinner("📝 Se corectează tema..."):
                    prompt = get_homework_correction_prompt(hw_materie, hw_text, from_photo=False)
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(
                            materie=MATERII.get(hw_materie),
                            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                            mod_avansat=st.session_state.get("mod_avansat", False),
                            mod_strategie=st.session_state.get("mod_strategie", False),
                            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                        )
                    ))
                st.session_state.hw_corectare = corectare
                st.session_state.hw_done      = True
                st.rerun()

    else:
        mat = st.session_state.get("hw_materie", "")
        src = "📷 din fotografie" if st.session_state.get("hw_from_photo") else "✏️ scrisă manual"
        st.caption(f"{mat} · temă {src}")
        if st.session_state.get("hw_from_photo") and st.session_state.get("hw_text"):
            with st.expander("📄 Text extras din poză", expanded=False):
                st.text(st.session_state.hw_text)
        st.markdown(st.session_state.hw_corectare)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Corectează altă temă", type="primary", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("hw_")]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("hw_")]:
                    st.session_state.pop(k, None)
                st.session_state.pop("homework_mode", None)
                st.rerun()


# === MOD QUIZ ===
NIVELE_QUIZ = ["🟢 Ușor (gimnaziu)", "🟡 Mediu (liceu)", "🔴 Greu (BAC)"]

MATERII_QUIZ = [m for m in list(MATERII.keys()) if m != "🤖 Automat"]


def get_quiz_prompt(materie_label: str, nivel: str, materie_val: str) -> str:
    """Generează prompt pentru crearea unui quiz."""
    nivel_text = nivel.split(" ", 1)[1].strip("()")
    return f"""Generează un quiz de 5 întrebări la {materie_label} pentru nivel {nivel_text}.

REGULI STRICTE:
1. Generează EXACT 5 întrebări numerotate (1. 2. 3. 4. 5.)
2. Fiecare întrebare are 4 variante de răspuns: A) B) C) D)
3. La finalul TUTUROR întrebărilor adaugă un bloc special cu răspunsurile corecte:

[[RASPUNSURI_CORECTE]]
1: X
2: X
3: X
4: X
5: X
[[/RASPUNSURI_CORECTE]]

unde X este A, B, C sau D.
4. Întrebările trebuie să fie clare și potrivite pentru nivel {nivel_text}.
5. Folosește LaTeX ($...$) pentru formule matematice.
6. NU da explicații acum — doar întrebările și răspunsurile corecte la final."""


def parse_quiz_response(response: str) -> tuple[str, dict]:
    """Extrage intrebarile si raspunsurile corecte din raspunsul AI.

    FIX: Gestioneaza corect cazurile cand AI-ul nu respecta exact delimitatorii:
    - Delimitatori lipsa: fallback prin cautarea unui bloc de raspunsuri
    - Formate variate: '1: A', '1. A', '1) A', '**1**: A'
    - Raspunsuri cu text extra: '1: A) text' -> extrage doar litera
    """
    correct = {}
    clean_response = response

    # Incearca mai intai delimitatorii exacti
    match = re.search(r'\[\[RASPUNSURI_CORECTE\]\](.*?)\[\[/RASPUNSURI_CORECTE\]\]',
                      response, re.DOTALL)

    # FIX: Fallback — AI-ul uneori omite delimitatorii sau ii scrie diferit
    if not match:
        match = re.search(
            r'(?:raspunsuri\s*corecte|raspunsuri\s*corecte|answers?)[:\s]*\n'
            r'((?:\s*\d+\s*[:.)-]\s*[A-D].*\n?){3,})',
            response, re.IGNORECASE | re.DOTALL
        )

    if match:
        block_start = match.start()
        clean_response = response[:block_start].strip()
        raw_block = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)

        for line in raw_block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # FIX: accepta formate: '1: A', '1. A', '1) A', '**1**: A', '1: A) text...'
            # FIX: regex mai strict — maxim 1 cifra pentru nr intrebare (evita "11: A" etc.)
            m = re.match(r'\*{0,2}(\d{1,2})\*{0,2}\s*[:.)-]\s*\*{0,2}([A-D])\b', line, re.IGNORECASE)
            if m:
                try:
                    q_num = int(m.group(1))
                    ans = m.group(2).upper()
                    correct[q_num] = ans
                except ValueError:
                    pass

    # FIX: Daca tot nu avem raspunsuri, incearca extractie din textul intreg
    if not correct:
        for m in re.finditer(
            r'(?:intrebarea|intrebarea|question)?\s*(\d+).*?'
            r'r[a]spuns(?:ul)?\s*(?:corect)?\s*[:\s]+([A-D])\b',
            response, re.IGNORECASE
        ):
            try:
                q_num = int(m.group(1))
                ans = m.group(2).upper()
                if 1 <= q_num <= 10:
                    correct[q_num] = ans
            except ValueError:
                pass

    return clean_response, correct


def evaluate_quiz(user_answers: dict, correct_answers: dict) -> tuple[int, str]:
    """Evaluează răspunsurile și returnează (scor, feedback_text)."""
    score = sum(1 for q, a in user_answers.items() if correct_answers.get(q) == a)
    total = len(correct_answers)

    lines = []
    for q in sorted(correct_answers.keys()):
        user_ans = user_answers.get(q, "—")
        correct_ans = correct_answers[q]
        if user_ans == correct_ans:
            lines.append(f"✅ **Întrebarea {q}**: {user_ans} — Corect!")
        else:
            lines.append(f"❌ **Întrebarea {q}**: ai răspuns **{user_ans}**, corect era **{correct_ans}**")

    if score == total:
        verdict = "🏆 Excelent! Nota 10!"
    elif score >= total * 0.8:
        verdict = "🌟 Foarte bine!"
    elif score >= total * 0.6:
        verdict = "👍 Bine, mai exersează puțin!"
    elif score >= total * 0.4:
        verdict = "📚 Trebuie să mai studiezi."
    else:
        verdict = "💪 Nu-ți face griji, încearcă din nou!"

    feedback = f"### Rezultat: {score}/{total} — {verdict}\n\n" + "\n\n".join(lines)
    return score, feedback


def run_quiz_ui():
    """Randează UI-ul pentru modul Quiz."""
    st.subheader("📝 Mod Examinare")

    # --- Setup quiz ---
    if not st.session_state.get("quiz_active"):
        col1, col2 = st.columns(2)
        with col1:
            quiz_materie_label = st.selectbox(
                "Materie:",
                options=MATERII_QUIZ,
                key="quiz_materie_select"
            )
        with col2:
            quiz_nivel = st.selectbox(
                "Nivel:",
                options=NIVELE_QUIZ,
                key="quiz_nivel_select"
            )

        if st.button("🚀 Generează Quiz", type="primary", use_container_width=True):
            quiz_materie_val = MATERII[quiz_materie_label]
            with st.spinner("📝 Profesorul pregătește întrebările..."):
                prompt = get_quiz_prompt(quiz_materie_label, quiz_nivel, quiz_materie_val)
                full_resp = ""
                for chunk in run_chat_with_rotation(
                    [], [prompt],
                    system_prompt=get_system_prompt(
                        materie=quiz_materie_val,
                        pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                        mod_avansat=st.session_state.get("mod_avansat", False),
                        mod_strategie=st.session_state.get("mod_strategie", False),
                        mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                    )
                ):
                    full_resp += chunk

            questions_text, correct = parse_quiz_response(full_resp)
            if len(correct) >= 3:
                st.session_state.quiz_active = True
                st.session_state.quiz_questions = questions_text
                st.session_state.quiz_correct = correct
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_materie = quiz_materie_label
                st.session_state.quiz_nivel = quiz_nivel
                st.rerun()
            else:
                st.error("❌ Nu am putut genera quiz-ul. Încearcă din nou.")
        return

    # --- Quiz activ ---
    st.caption(f"📚 {st.session_state.quiz_materie} · {st.session_state.quiz_nivel}")

    # Afișează întrebările
    st.markdown(st.session_state.quiz_questions)
    st.divider()

    if not st.session_state.quiz_submitted:
        st.markdown("**Alege răspunsurile tale:**")
        answers = {}
        for q_num in sorted(st.session_state.quiz_correct.keys()):
            answers[q_num] = st.radio(
                f"Întrebarea {q_num}:",
                options=["A", "B", "C", "D"],
                horizontal=True,
                key=f"quiz_ans_{q_num}",
                index=None
            )

        all_answered = all(v is not None for v in answers.values())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Trimite răspunsurile", type="primary",
                         disabled=not all_answered, use_container_width=True):
                st.session_state.quiz_answers = {k: v for k, v in answers.items() if v}
                st.session_state.quiz_submitted = True
                st.rerun()
        with col2:
            if st.button("🔄 Quiz nou", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
    else:
        # Afișează rezultatele
        score, feedback = evaluate_quiz(
            st.session_state.quiz_answers,
            st.session_state.quiz_correct
        )
        st.markdown(feedback)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Quiz nou", type="primary", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted", "quiz_mode"]:
                    st.session_state.pop(k, None)
                st.rerun()



# ============================================================
# === GROQ CLIENT ===
# Groq nu are context caching — system prompt-ul e trimis la fiecare apel.
# Compensație: Groq e mult mai rapid decât Gemini (răspunsuri în <1s).

def run_chat_with_rotation(history_obj, payload, system_prompt=None):
    """Rulează chat cu rotație automată a cheilor API Groq.

    Compatibil cu același API ca versiunea Gemini — history_obj și payload
    au același format, returnează un generator de chunks text.
    """
    GROQ_MODEL = "llama-3.3-70b-versatile"

    if not keys:
        raise Exception(
            "Nicio cheie API Groq configurată. "
            "Adaugă cel puțin o cheie în st.secrets['GOOGLE_API_KEYS'] sau introdu-o manual în sidebar."
        )

    active_prompt = system_prompt or st.session_state.get("system_prompt") or SYSTEM_PROMPT
    max_retries = max(len(keys) * 3, 10)  # minim 10 încercări
    last_error = None
    _rate_limit_attempts = 0  # contor separat pentru 429

    for attempt in range(max_retries):
        if st.session_state.key_index >= len(keys):
            st.session_state.key_index = 0
        current_key = keys[st.session_state.key_index]

        try:
            client = GroqClient(api_key=current_key)

            # Construim istoricul în format OpenAI-compatibil (același ca Groq)
            messages = [{"role": "system", "content": active_prompt}]
            for msg in history_obj:
                role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
                parts = msg.get("parts", [])
                if isinstance(parts, list):
                    text_parts = []
                    for p in parts:
                        if isinstance(p, str) and p.startswith("data:image/"):
                            text_parts.append("[imagine atașată anterior]")
                        elif isinstance(p, str):
                            text_parts.append(p)
                        else:
                            text_parts.append(str(p))
                    content = " ".join(text_parts)
                else:
                    content = str(parts)
                if content.strip():
                    messages.append({"role": role, "content": content})

            # Detectăm dacă payload-ul conține imagini
            payload_list = payload if isinstance(payload, list) else [payload]
            has_image = any(isinstance(p, str) and p.startswith("data:image/") for p in payload_list)

            # Selectăm modelul: vision dacă avem imagini, text dacă nu
            model_to_use = "meta-llama/llama-4-scout-17b-16e-instruct" if has_image else GROQ_MODEL

            # Adăugăm mesajul curent (payload)
            user_parts = []
            for p in payload_list:
                if isinstance(p, str) and p.startswith("data:image/"):
                    user_parts.append({"type": "image_url", "image_url": {"url": p}})
                elif isinstance(p, str):
                    user_parts.append({"type": "text", "text": p})
                else:
                    user_parts.append({"type": "text", "text": str(p)})

            if has_image:
                # Model vision: trimitem array cu imagine + text
                messages.append({"role": "user", "content": user_parts})
            else:
                # Model text: string simplu
                messages.append({"role": "user", "content": " ".join(p["text"] for p in user_parts)})

            # Apel streaming
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=4096,
                temperature=0.7,
                stream=True,
            )

            _output_tokens = 0
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    _output_tokens += len(delta.split())  # estimare
                    yield delta

            # Actualizăm contoarele token per cheie
            _key_id = f"_tokens_key_{st.session_state.get('key_index', 0)}"
            _prev = st.session_state.get(_key_id, {"prompt": 0, "output": 0, "calls": 0})
            st.session_state[_key_id] = {
                "prompt": _prev["prompt"],
                "output": _prev["output"] + _output_tokens,
                "calls":  _prev["calls"] + 1,
            }
            return

        except Exception as e:
            last_error = e
            error_msg = str(e) + " " + repr(e)

            _is_rate_limit = (
                "429" in error_msg
                or "rate_limit" in error_msg.lower()
                or "rate limit" in error_msg.lower()
                or "too many requests" in error_msg.lower()
            )
            _is_invalid_key = (
                "invalid_api_key" in error_msg.lower()
                or "authentication" in error_msg.lower()
                or "401" in error_msg
                or ("403" in error_msg and "quota" not in error_msg.lower())
            )
            _is_quota = (
                "quota" in error_msg.lower()
                and "429" not in error_msg  # quota zilnică, nu rate limit pe minut
            )

            if _is_rate_limit:
                # Rate limit pe minut (429) — backoff exponențial
                _rate_limit_attempts += 1
                if _rate_limit_attempts == 1:
                    wait = 5
                elif _rate_limit_attempts == 2:
                    wait = 10
                elif _rate_limit_attempts == 3:
                    wait = 20
                else:
                    wait = 30
                st.toast(f"⏳ Limită Groq depășită — reîncerc în {wait}s... ({_rate_limit_attempts}/4)", icon="🔄")
                time.sleep(wait)
                if _rate_limit_attempts >= 4:
                    raise Exception(
                        "Groq rate limit activ. Ai trimis prea multe mesaje într-un minut. "
                        "Așteaptă 30-60 de secunde și încearcă din nou. 🕐"
                    )
                continue

            elif _is_invalid_key or _is_quota:
                # Cheie invalidă sau quota zilnică epuizată — rotăm la altă cheie
                _quota_key = "_quota_rotations"
                rotations = st.session_state.get(_quota_key, 0) + 1
                st.session_state[_quota_key] = rotations
                if len(keys) <= 1 or rotations >= len(keys):
                    st.session_state.pop(_quota_key, None)
                    raise Exception(
                        "Cheia API Groq este invalidă sau quota zilnică s-a epuizat. "
                        "Verifică cheia în console.groq.com sau adaugă o cheie nouă în sidebar. 🔑"
                    )
                st.toast(f"⚠️ Cheie invalidă — schimb la cheia {st.session_state.key_index + 2}...", icon="🔄")
                st.session_state.key_index = (st.session_state.key_index + 1) % len(keys)
                time.sleep(0.5)
                continue

            elif "503" in error_msg or "overloaded" in error_msg.lower():
                wait = min(0.5 * (2 ** attempt), 5)
                st.toast("🐢 Server ocupat, reîncerc...", icon="⏳")
                time.sleep(wait)
                continue

            else:
                raise e

    st.session_state.pop("_quota_rotations", None)
    raise Exception(
        "Groq rate limit activ — ai trimis prea multe mesaje rapid. "
        "Așteaptă 30-60 de secunde și încearcă din nou. 🕐"
    )

# === UI PRINCIPAL ===
st.title("🎓 Profesor Liceu · Groq")

# Afișăm materia selectată mic sub titlu
if st.session_state.get("pedagogie_mode"):
    st.caption("🧠 **Mod Sfaturi de studiu**")
else:
    _mat_curenta = st.session_state.get("materie_selectata")
    if _mat_curenta:
        _mat_label = next((k for k, v in MATERII.items() if v == _mat_curenta), _mat_curenta)
        st.caption(f"Materie selectată: **{_mat_label}**")

with st.sidebar:
    st.header("⚙️ Opțiuni")

    # --- Selector materie ---
    st.subheader("📚 Materie")
    _materii_keys = list(MATERII.keys())
    _mat_saved = st.session_state.get("materie_selectata")
    _mat_default_idx = next(
        (i for i, k in enumerate(_materii_keys) if MATERII[k] == _mat_saved),
        0  # fallback la "🤖 Automat" dacă nu găsim
    )
    materie_label = st.selectbox(
        "Alege materia:",
        options=_materii_keys,
        index=_mat_default_idx,
        label_visibility="collapsed"
    )
    materie_selectata = MATERII[materie_label]
    _mod_automat = (materie_selectata is None)  # True când e "🤖 Automat"

    # Actualizează system prompt dacă s-a schimbat materia
    if st.session_state.get("materie_selectata") != materie_selectata:
        st.session_state.materie_selectata = materie_selectata
        if _mod_automat:
            # Mod automat — resetăm detecția, promptul va fi setat la primul mesaj
            st.session_state.pop("_detected_subject", None)
            st.session_state.pop("_pending_user_msg", None)
            st.session_state.system_prompt = get_system_prompt(
                materie=None,
                pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                mod_avansat=st.session_state.get("mod_avansat", False),
                mod_strategie=st.session_state.get("mod_strategie", False),
                mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
            )
        else:
            # Mod manual — selectorul are prioritate absolută
            st.session_state["_detected_subject"] = materie_selectata
            st.session_state.pop("_pending_user_msg", None)
            st.session_state.system_prompt = get_system_prompt(
                materie_selectata,
                pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                mod_avansat=st.session_state.get("mod_avansat", False),
                mod_strategie=st.session_state.get("mod_strategie", False),
                mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
            )
        # Forțăm rerun explicit — necesar pe mobil unde sidebar-ul nu declanșează
        # automat rerender-ul paginii principale după schimbare de materie
        st.rerun()

    # Info materie curentă sub selector
    if _mod_automat:
        _detected_now = st.session_state.get("_detected_subject")
        if _detected_now and _detected_now != "pedagogie":
            _det_label = _MATERII_LABEL.get(_detected_now, _detected_now.capitalize())
            st.caption(f"🔍 Detectat: **{_det_label}**")
        elif not _detected_now:
            st.caption("🔍 Materia se detectează automat din mesaj")
    else:
        st.info(f"Focusat pe: **{materie_label}**")

    # --- Toggle Sfaturi de studiu ---
    # Când se activează: salvează sesiunea curentă și deschide conversație nouă dedicată.
    # Când se dezactivează: restaurează sesiunea anterioară (sau meniul principal dacă nu exista).
    _ped_active = st.session_state.get("pedagogie_mode", False)
    _ped_toggle = st.toggle(
        "🧠 Sfaturi de studiu",
        value=_ped_active,
        help="Activează pentru sfaturi de organizare și tehnici de învățare eficientă. Dezactivează pentru a reveni la profesor."
    )

    if _ped_toggle != _ped_active:
        if _ped_toggle:
            # ── ACTIVARE: salvăm sesiunea curentă și deschidem una nouă ──
            st.session_state["_ped_prev_session_id"]   = st.session_state.get("session_id", "")
            st.session_state["_ped_prev_messages"]     = list(st.session_state.get("messages", []))
            st.session_state["_ped_prev_materie"]      = st.session_state.get("materie_selectata")
            st.session_state["_ped_prev_detected"]     = st.session_state.get("_detected_subject")
            st.session_state["_ped_prev_system_prompt"]= st.session_state.get("system_prompt", "")

            # Sesiune nouă dedicată sfaturilor de studiu
            _ped_sid = generate_unique_session_id()
            register_session(_ped_sid)
            st.session_state["session_id"] = _ped_sid
            st.session_state["messages"]   = []
            # FIX 1: adăugăm sesiunea de pedagogie în lista locală — apare în sidebar
            _my_sids = st.session_state.get("_my_session_ids", [])
            if _ped_sid not in _my_sids:
                _my_sids.append(_ped_sid)
            st.session_state["_my_session_ids"] = _my_sids
            # Curățăm modurile active (BAC, temă, quiz)
            for _k in ["bac_mode", "bac_active", "bac_materie", "bac_profil", "bac_subject",
                       "bac_barem", "bac_raspuns", "bac_corectat", "bac_corectare",
                       "bac_start_time", "bac_timp_min", "bac_from_photo", "bac_ocr_done",
                       "bac_timer_submitted", "homework_mode", "hw_materie", "hw_text",
                       "hw_corectare", "hw_done", "hw_from_photo", "hw_ocr_done",
                       "quiz_mode", "quiz_active", "quiz_questions", "quiz_correct",
                       "quiz_answers", "quiz_submitted", "quiz_materie", "quiz_nivel",
                       "_suggested_question", "_pending_user_msg"]:
                st.session_state.pop(_k, None)
            st.session_state["pedagogie_mode"]    = True
            st.session_state["_detected_subject"] = "pedagogie"
            st.session_state["system_prompt"]     = get_system_prompt(
                materie="pedagogie",
                pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                mod_avansat=st.session_state.get("mod_avansat", False),
                mod_strategie=st.session_state.get("mod_strategie", False),
                mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
            )
            invalidate_session_cache()
            components.html(
                f"<script>localStorage.setItem('profesor_session_id', {json.dumps(_ped_sid)});</script>",
                height=0,
            )
        else:
            # ── DEZACTIVARE: restaurăm sesiunea anterioară ──
            _prev_sid = st.session_state.get("_ped_prev_session_id", "")
            _prev_msg = st.session_state.get("_ped_prev_messages", [])
            _prev_mat = st.session_state.get("_ped_prev_materie")
            _prev_det = st.session_state.get("_ped_prev_detected")
            _prev_sys = st.session_state.get("_ped_prev_system_prompt", "")

            st.session_state["pedagogie_mode"] = False
            # Curățăm cheile temporare de salvare
            for _k in ["_ped_prev_session_id", "_ped_prev_messages",
                       "_ped_prev_materie", "_ped_prev_detected", "_ped_prev_system_prompt"]:
                st.session_state.pop(_k, None)

            if _prev_sid and is_valid_session_id(_prev_sid):
                # Restaurăm sesiunea anterioară
                st.session_state["session_id"]        = _prev_sid
                st.session_state["messages"]          = _prev_msg
                st.session_state["materie_selectata"] = _prev_mat
                st.session_state["_detected_subject"] = _prev_det
                st.session_state["system_prompt"]     = _prev_sys or get_system_prompt(
                    materie=_prev_mat,
                    pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                    mod_avansat=st.session_state.get("mod_avansat", False),
                    mod_strategie=st.session_state.get("mod_strategie", False),
                    mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                )
                # FIX 3: actualizăm și URL-ul ?sid= — altfel la refresh se restaurează SID-ul de pedagogie
                try:
                    st.query_params["sid"] = _prev_sid
                except Exception:
                    pass
                components.html(
                    f"<script>localStorage.setItem('profesor_session_id', {json.dumps(_prev_sid)});</script>",
                    height=0,
                )
            else:
                # Nu exista sesiune anterioară → meniu principal (ecran curat)
                _new_main_sid = generate_unique_session_id()
                register_session(_new_main_sid)
                st.session_state["session_id"]        = _new_main_sid
                st.session_state["messages"]          = []
                st.session_state["materie_selectata"] = None
                st.session_state.pop("_detected_subject", None)
                st.session_state["system_prompt"]     = get_system_prompt(
                    materie=None,
                    pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                    mod_avansat=st.session_state.get("mod_avansat", False),
                    mod_strategie=st.session_state.get("mod_strategie", False),
                    mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                )
                # FIX 3b: actualizăm URL-ul și localStorage la sesiunea nouă
                try:
                    st.query_params["sid"] = _new_main_sid
                except Exception:
                    pass
                components.html(
                    f"<script>localStorage.setItem('profesor_session_id', {json.dumps(_new_main_sid)});</script>",
                    height=0,
                )
            invalidate_session_cache()
        st.rerun()

    st.divider()

    # --- Dark Mode toggle ---
    dark_mode = st.toggle("🌙 Mod Întunecat", value=st.session_state.get("dark_mode", False))
    if dark_mode != st.session_state.get("dark_mode", False):
        st.session_state.dark_mode = dark_mode
        st.rerun()

    # --- Mod Pas cu Pas ---
    pas_cu_pas = st.toggle(
        "🔢 Explicație Pas cu Pas",
        value=st.session_state.get("pas_cu_pas", False),
        help="Profesorul va explica fiecare problemă detaliat, pas cu pas, cu motivația fiecărei operații."
    )
    if pas_cu_pas != st.session_state.get("pas_cu_pas", False):
        st.session_state.pas_cu_pas = pas_cu_pas
        # Regenerează prompt-ul cu noul mod
        st.session_state.system_prompt = get_system_prompt(
            materie=st.session_state.get("materie_selectata"),
            pas_cu_pas=pas_cu_pas,
            mod_avansat=st.session_state.get("mod_avansat", False),
            mod_strategie=st.session_state.get("mod_strategie", False),
            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
        )
        if pas_cu_pas:
            st.toast("🔢 Mod Pas cu Pas activat!", icon="✅")
        else:
            st.toast("Mod normal activat.", icon="💬")
        st.rerun()

    if st.session_state.get("pas_cu_pas"):
        st.info("🔢 **Pas cu Pas activ** — fiecare problemă e explicată detaliat.", icon="📋")

    # --- Mod Explică-mi Strategia ---
    mod_strategie = st.toggle(
        "🧠 Explică-mi Strategia",
        value=st.session_state.get("mod_strategie", False),
        help="Profesorul explică CUM să gândești rezolvarea — logica și strategia, nu calculele."
    )
    if mod_strategie != st.session_state.get("mod_strategie", False):
        st.session_state.mod_strategie = mod_strategie
        st.session_state.system_prompt = get_system_prompt(
            st.session_state.get("materie_selectata"),
            mod_avansat=st.session_state.get("mod_avansat", False),
            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
            mod_strategie=mod_strategie,
            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False)
        )
        st.toast("🧠 Mod Strategie activat!" if mod_strategie else "Mod normal activat.", icon="✅" if mod_strategie else "💬")
        st.rerun()
    if st.session_state.get("mod_strategie"):
        st.info("🧠 **Strategie activ** — înveți să gândești, nu să copiezi.", icon="🗺️")

    # --- Mod Avansat ---
    mod_avansat = st.toggle(
        "⚡ Mod Avansat",
        value=st.session_state.get("mod_avansat", False),
        help="Știi deja bazele? Profesorul sare peste explicații evidente și îți dă doar ideea cheie și calculul esențial."
    )
    if mod_avansat != st.session_state.get("mod_avansat", False):
        st.session_state.mod_avansat = mod_avansat
        st.session_state.system_prompt = get_system_prompt(
            st.session_state.get("materie_selectata"),
            mod_avansat=mod_avansat,
            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
            mod_strategie=st.session_state.get("mod_strategie", False),
            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
        )
        st.toast("⚡ Mod Avansat activat!" if mod_avansat else "Mod normal activat.", icon="✅" if mod_avansat else "💬")
        st.rerun()
    if st.session_state.get("mod_avansat"):
        st.info("⚡ **Mod Avansat activ** — răspunsuri scurte, doar esențialul.", icon="🎯")

    # --- Mod Pregătire BAC Intensivă ---
    mod_bac_intensiv = st.toggle(
        "🎓 Pregătire BAC Intensivă",
        value=st.session_state.get("mod_bac_intensiv", False),
        help="Focusat pe ce pică la BAC: tipare de subiecte, punctaj, timp, teorie lipsă detectată automat."
    )
    if mod_bac_intensiv != st.session_state.get("mod_bac_intensiv", False):
        st.session_state.mod_bac_intensiv = mod_bac_intensiv
        st.session_state.system_prompt = get_system_prompt(
            st.session_state.get("materie_selectata"),
            mod_avansat=st.session_state.get("mod_avansat", False),
            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
            mod_strategie=st.session_state.get("mod_strategie", False),
            mod_bac_intensiv=mod_bac_intensiv
        )
        st.toast("🎓 Mod BAC Intensiv activat!" if mod_bac_intensiv else "Mod normal activat.", icon="✅" if mod_bac_intensiv else "💬")
        st.rerun()
    if st.session_state.get("mod_bac_intensiv"):
        st.info("🎓 **BAC Intensiv activ** — focusat pe ce pică la examen.", icon="📝")

    st.divider()

    # --- Status Supabase ---
    if not st.session_state.get("_sb_online", True):
        st.markdown(
            '<div style="background:#e67e22;color:white;padding:8px 12px;'
            'border-radius:8px;font-size:13px;text-align:center;margin-bottom:8px">'
            '📴 Mod offline — datele sunt salvate local</div>',
            unsafe_allow_html=True
        )
    else:
        pending = len(st.session_state.get("_offline_queue", []))
        if pending:
            st.caption(f"☁️ {pending} mesaje în așteptare pentru sincronizare")


    st.divider()

    if st.button("🗑️ Șterge Istoricul", type="primary"):
        clear_history_db(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("📁 Materiale")

    # Groq suportă imagini via base64 (vision). PDF-urile sunt citite ca text.
    uploaded_file = st.file_uploader(
        "Încarcă imagine sau PDF",
        type=["jpg", "jpeg", "png", "webp", "gif", "pdf"],
        help="Imaginile sunt trimise la AI în format base64. PDF-urile sunt citite ca text."
    )
    media_content = None  # bytes sau text — trimis la AI

    if uploaded_file and st.session_state.get("_removed_file_key") == f"{uploaded_file.name}_{uploaded_file.size}":
        uploaded_file = None

    if uploaded_file:
        st.session_state.pop("_removed_file_key", None)
        file_key  = f"_gfile_{uploaded_file.name}_{uploaded_file.size}"
        file_type = uploaded_file.type
        is_image  = file_type.startswith("image/")

        # Cache bytes în session_state — evităm re-citirea la fiecare rerun
        if file_key not in st.session_state:
            st.session_state[file_key] = uploaded_file.getvalue()

        media_content = st.session_state[file_key]

        st.session_state["_current_uploaded_file_meta"] = {
            "name": uploaded_file.name,
            "type": file_type,
            "size": uploaded_file.size,
        }

        if is_image:
            st.image(uploaded_file, caption=f"🖼️ {uploaded_file.name}", use_container_width=True)
            st.success("✅ Imaginea e pregătită — AI-ul o analizează vizual.")
        else:
            st.success(f"✅ **{uploaded_file.name}** încărcat ({uploaded_file.size // 1024} KB)")
            st.caption("📄 AI-ul poate citi și analiza conținutul documentului.")

        if st.button("🗑️ Elimină fișierul", use_container_width=True, key="remove_media"):
            st.session_state.pop(file_key, None)
            media_content = None
            st.session_state.pop("_current_uploaded_file_meta", None)
            st.session_state["_removed_file_key"] = f"{uploaded_file.name}_{uploaded_file.size}"
            st.rerun()

    st.divider()

    # --- Mod Quiz + BAC ---
    st.subheader("📝 Examinare & BAC")

    # Chei exacte per mod — actualizați când adăugați chei noi în fiecare mod
    _BAC_KEYS = [
        "bac_mode", "bac_active", "bac_materie", "bac_profil", "bac_subject",
        "bac_barem", "bac_raspuns", "bac_corectat", "bac_corectare",
        "bac_start_time", "bac_timp_min", "bac_from_photo", "bac_ocr_done",
        "bac_timer_submitted",
    ]
    _HW_KEYS = [
        "homework_mode", "hw_materie", "hw_text", "hw_corectare",
        "hw_done", "hw_from_photo", "hw_ocr_done",
    ]
    _QUIZ_KEYS = [
        "quiz_mode", "quiz_active", "quiz_questions", "quiz_correct",
        "quiz_answers", "quiz_submitted", "quiz_materie", "quiz_nivel",
    ]
    _SHARED_KEYS = ["_suggested_question", "_pending_user_msg"]  # FIX 2: pedagogie_mode eliminat — nu trebuie șters de Quiz/BAC/Temă

    def _clear_all_modes():
        for k in _BAC_KEYS + _HW_KEYS + _QUIZ_KEYS + _SHARED_KEYS:
            st.session_state.pop(k, None)

    col_q, col_b = st.columns(2)
    with col_q:
        if st.button("🎯 Quiz rapid", use_container_width=True,
                     type="primary" if st.session_state.get("quiz_mode") else "secondary"):
            entering = not st.session_state.get("quiz_mode", False)
            _clear_all_modes()
            st.session_state.quiz_mode = entering
            st.session_state.pop("bac_mode", None)
            st.session_state.pop("homework_mode", None)
            st.rerun()
    with col_b:
        if st.button("🎓 Simulare BAC", use_container_width=True,
                     type="primary" if st.session_state.get("bac_mode") else "secondary"):
            entering = not st.session_state.get("bac_mode", False)
            _clear_all_modes()
            st.session_state.bac_mode = entering
            st.session_state.pop("quiz_mode", None)
            st.session_state.pop("homework_mode", None)
            st.rerun()

    if st.button("📚 Corectează Temă", use_container_width=True,
                 type="primary" if st.session_state.get("homework_mode") else "secondary"):
        entering = not st.session_state.get("homework_mode", False)
        _clear_all_modes()
        st.session_state.homework_mode = entering
        st.session_state.pop("quiz_mode", None)
        st.session_state.pop("bac_mode", None)
        st.rerun()

    st.divider()

    # --- Istoric conversații ---
    st.subheader("🕐 Conversații anterioare")
    if st.button("🔄 Conversație nouă", use_container_width=True):
        _cleanup_gfiles()
        new_sid = generate_unique_session_id()
        register_session(new_sid)
        # Salvează noul SID în lista sesiunilor acestui browser
        _my_sids = st.session_state.get("_my_session_ids", [])
        if new_sid not in _my_sids:
            _my_sids.append(new_sid)
        st.session_state["_my_session_ids"] = _my_sids
        switch_session(new_sid)
        st.rerun()

    # Afișează DOAR sesiunile create de acest browser în această sesiune Streamlit
    # (nu toate sesiunile din Supabase — acelea aparțin altor utilizatori)
    current_sid = st.session_state.session_id
    _my_sids = st.session_state.get("_my_session_ids", [current_sid])
    if current_sid not in _my_sids:
        _my_sids = [current_sid] + _my_sids
        st.session_state["_my_session_ids"] = _my_sids

    # Încarcă preview-urile doar pentru sesiunile acestui browser
    sessions = []
    try:
        _supabase = get_supabase_client()
        _resp = (
            _supabase.table("session_previews")
            .select("session_id, last_active, msg_count, preview")
            .eq("app_id", get_app_id())
            .in_("session_id", _my_sids)
            .gt("msg_count", 0)
            .order("last_active", desc=True)
            .limit(15)
            .execute()
        )
        sessions = _resp.data or []
    except Exception:
        pass

    for s in sessions:
        is_current = s["session_id"] == current_sid
        # FIX 5: etichetă vizuală pentru sesiunile de sfaturi de studiu
        _preview_text = s['preview'] or "Conversație"
        _is_ped_session = _preview_text.lower().startswith(("sfat", "studi", "tehnic", "înv", "inv", "📚", "🧠"))
        _ped_prefix = "🧠 " if _is_ped_session else ""
        label = f"{'▶ ' if is_current else ''}{_ped_prefix}{_preview_text}"
        caption = f"{format_time_ago(s['last_active'])} · {s['msg_count']} mesaje"
        with st.container():
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    label,
                    key=f"sess_{s['session_id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    help=caption,
                ):
                    if not is_current:
                        switch_session(s["session_id"])
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{s['session_id']}", help="Șterge"):
                    clear_history_db(s["session_id"])
                    if is_current:
                        st.session_state.messages = []
                    # Scoate din lista locală
                    _my_sids2 = st.session_state.get("_my_session_ids", [])
                    if s["session_id"] in _my_sids2:
                        _my_sids2.remove(s["session_id"])
                    st.session_state["_my_session_ids"] = _my_sids2
                    st.rerun()

    st.divider()

    _debug_val = st.session_state.get("_debug_info_open", False)
    _debug_checked = st.checkbox("🔧 Debug Info", value=_debug_val, key="chk_debug_info")
    if _debug_checked != _debug_val:
        st.session_state["_debug_info_open"] = _debug_checked

    if _debug_checked:
        msg_count = len(st.session_state.get("messages", []))
        st.caption(f"📊 Mesaje în memorie: {msg_count}/{MAX_MESSAGES_IN_MEMORY}")
        st.caption(f"🔑 Cheie API activă: {st.session_state.key_index + 1}/{len(keys)}")

        # ── Statistici token usage per cheie (sesiunea curentă) ──
        # Notă: Gemini Free tier = 1.500 req/zi și 1.000.000 token/min per cheie.
        # Nu avem acces la quota rămasă prin API — afișăm consumul din sesiunea curentă.
        _active_idx = st.session_state.get("key_index", 0)
        _key_id = f"_tokens_key_{_active_idx}"
        _usage = st.session_state.get(_key_id, {"prompt": 0, "output": 0, "calls": 0})
        _total_tok = _usage["prompt"] + _usage["output"]
        _calls = _usage["calls"]
        if _calls > 0:
            st.caption(f"📈 Tokeni folosiți (cheia {_active_idx + 1}, sesiunea curentă):")
            st.caption(f"   ↳ Input: {_usage['prompt']:,} · Output: {_usage['output']:,} · Total: {_total_tok:,}")
            st.caption(f"   ↳ Apeluri AI: {_calls} · Medie/apel: {_total_tok // max(_calls,1):,} tok")
            # Bară vizuală față de limita de 1M tokeni/minut (limita de rate, nu de quota zilnică)
            _pct = min(_total_tok / 1_000_000 * 100, 100)
            _bar_filled = int(_pct / 5)
            _bar = "█" * _bar_filled + "░" * (20 - _bar_filled)
            _color = "🟢" if _pct < 50 else ("🟡" if _pct < 80 else "🔴")
            st.caption(f"   {_color} [{_bar}] {_pct:.1f}% din 1M tok/min")
        else:
            st.caption("📈 Tokeni folosiți: 0 (niciun apel AI în sesiunea curentă)")

        # Sumar pentru toate cheile din sesiune
        _all_keys_usage = []
        for i in range(len(keys)):
            _u = st.session_state.get(f"_tokens_key_{i}", {"prompt": 0, "output": 0, "calls": 0})
            if _u["calls"] > 0:
                _all_keys_usage.append(f"Cheia {i+1}: {_u['prompt']+_u['output']:,} tok ({_u['calls']} apeluri)")
        if len(_all_keys_usage) > 1:
            st.caption("📋 Toate cheile folosite: " + " | ".join(_all_keys_usage))

        st.caption(f"🆔 Sesiune: {st.session_state.session_id[:16]}...")


# === MAIN UI — TEME / BAC / QUIZ / CHAT ===
if st.session_state.get("homework_mode"):
    run_homework_ui()
    st.stop()

if st.session_state.get("bac_mode"):
    run_bac_sim_ui()
    st.stop()

if st.session_state.get("quiz_mode"):
    run_quiz_ui()
    st.stop()

# === ÎNCĂRCARE MESAJE (CHAT MODE) ===
# Încărcăm istoricul dacă: nu există messages, sau messages aparțin altei sesiuni
_current_sid = st.session_state.session_id
if (
    "messages" not in st.session_state
    or st.session_state.get("_messages_for_sid") != _current_sid
):
    _loaded_msgs = load_history_from_db(_current_sid)
    st.session_state.messages = _loaded_msgs
    st.session_state["_messages_for_sid"] = _current_sid
    st.session_state.pop("_history_may_be_incomplete", None)

    # ── Revenire din altă sesiune/zi: pre-generăm rezumatul de context ──
    # FIX 7: nu generăm rezumat dacă lista e goală (sesiune nouă de pedagogie sau chat nou)
    _loaded_count = len(_loaded_msgs)
    if _loaded_count > MAX_MESSAGES_TO_SEND_TO_AI:
        _sum_key     = "_conversation_summary"
        _sum_sid_key = "_summary_for_sid"
        _needs_summary = (
            not st.session_state.get(_sum_key)
            or st.session_state.get(_sum_sid_key) != st.session_state.session_id
        )
        if _needs_summary:
            try:
                with st.spinner("📚 Profesorul reîncarcă contextul conversației anterioare..."):
                    _auto_summary = summarize_conversation(_loaded_msgs)
                if _auto_summary:
                    st.session_state[_sum_key]     = _auto_summary
                    st.session_state["_summary_cached_at"] = _loaded_count
                    st.session_state[_sum_sid_key] = st.session_state.session_id
                    st.toast("✅ Contextul conversației anterioare a fost reîncărcat!", icon="🧠")
            except Exception:
                # Rezumatul e opțional — dacă Groq dă 429 sau altă eroare la refresh,
                # ignorăm silențios. Conversația funcționează normal fără rezumat.
                pass

# Banner mod Pas cu Pas
if st.session_state.get("pas_cu_pas"):
    st.markdown(
        '<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;'
        'padding:10px 16px;border-radius:10px;margin-bottom:12px;'
        'display:flex;align-items:center;gap:10px;font-size:14px;">'
        '🔢 <strong>Mod Pas cu Pas activ</strong> — '
        'Profesorul îți va explica fiecare problemă detaliat, cu motivația fiecărui pas.'
        '</div>',
        unsafe_allow_html=True
    )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_with_svg(msg["content"])
        else:
            st.markdown(msg["content"])

    # Butoanele apar DOAR sub ultimul mesaj al profesorului
    if (msg["role"] == "assistant" and
            i == len(st.session_state.messages) - 1):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Nu am înțeles", key="qa_reexplain", use_container_width=True, help="Explică altfel, cu o altă analogie"):
                st.session_state["_quick_action"] = "reexplain"
                st.rerun()
        with col2:
            if st.button("✏️ Exercițiu similar", key="qa_similar", use_container_width=True, help="Generează un exercițiu similar pentru practică"):
                st.session_state["_quick_action"] = "similar"
                st.rerun()
        with col3:
            if st.button("🧠 Explică strategia", key="qa_strategy", use_container_width=True, help="Cum să gândești acest tip de problemă"):
                st.session_state["_quick_action"] = "strategy"
                st.rerun()


# ── Handler pentru butoanele de acțiuni rapide ──

TYPING_HTML = """
<div class="typing-indicator">
    <div class="typing-dots"><span></span><span></span><span></span></div>
    <span>Domnul Profesor scrie...</span>
</div>
"""

if st.session_state.get("_quick_action"):
    action = st.session_state.pop("_quick_action")
    # FIX Bug 2: _quick_action_ref nu era setat nicăieri — eliminat, nu mai e necesar
    # (context-ul vine direct din ultimul mesaj al asistentului/utilizatorului)

    # ── Găsește ultimul mesaj al asistentului pentru context real ──
    last_assistant_msg = ""
    last_user_msg = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and not last_assistant_msg:
            last_assistant_msg = msg["content"]
        if msg["role"] == "user" and not last_user_msg:
            last_user_msg = msg["content"]
        if last_assistant_msg and last_user_msg:
            break

    # FIX Bug 2: logică robustă pentru prev_topic și prev_question
    # - curățăm LaTeX (\$...\$, \$\$...\$\$) și markdown înainte de trunchiere
    # - trunchierea se face la spațiu, nu în mijlocul unui cuvânt LaTeX
    # - fallback explicit dacă mesajele lipsesc
    import re as _re
    _clean = lambda t: _re.sub(r'\$\$[\s\S]*?\$\$|\$[^\$\n]*?\$|[*`#\\]', '', t).strip()
    _clean2 = lambda t: _re.sub(r'\s+', ' ', _clean(t))  # colapsăm whitespace multiplu

    if last_assistant_msg:
        _cleaned = _clean2(last_assistant_msg)
        # Trunchierea la 120 de caractere, la granița unui cuvânt
        if len(_cleaned) > 120:
            prev_topic = _cleaned[:120].rsplit(' ', 1)[0].rstrip('.,;:') + "..."
        else:
            prev_topic = _cleaned or "subiectul anterior"
    else:
        prev_topic = "subiectul anterior"

    if last_user_msg:
        _cleaned_q = _clean2(last_user_msg)
        prev_question = _cleaned_q[:100] if len(_cleaned_q) > 100 else _cleaned_q
        prev_question = prev_question or "întrebarea anterioară"
    else:
        prev_question = "întrebarea anterioară"

    action_prompts = {
        "reexplain": (
            f"Nu am înțeles explicația ta despre: '{prev_topic}'. "
            f"Te rog să explici din nou, dar complet diferit — "
            f"altă analogie, altă ordine a pașilor, exemple mai simple din viața reală. "
            f"Evită exact aceleași cuvinte și structura anterioară."
        ),
        "similar": (
            (
                f"Generează un exercițiu similar cu '{prev_question}', "
                f"folosind alt cuvânt sau altă situație de comunicare, cu dificultate puțin mai mare. "
                f"Enunță exercițiul ÎNTÂI, apoi rezolvă-l complet pas cu pas."
            ) if st.session_state.get("materie_selectata") in ("limba engleză", "limba franceză", "limba germană")
            else (
                f"Generează un exercițiu similar cu '{prev_question}', "
                f"cu date numerice diferite și dificultate puțin mai mare. "
                f"Enunță exercițiul ÎNTÂI, apoi rezolvă-l complet pas cu pas."
            )
        ),
        "strategy": (
            f"Explică-mi STRATEGIA de gândire pentru '{prev_question}': "
            f"cum recunosc că e acest tip, ce fac primul pas în minte, ce capcane să evit. "
            f"Fără calcule — vreau doar logica și gândirea din spate."
        ),
    }
    injected = action_prompts.get(action, "")
    if injected:
        with st.chat_message("user"):
            st.markdown(injected)
        st.session_state.messages.append({"role": "user", "content": injected})
        save_message_with_limits(st.session_state.session_id, "user", injected)

        context_messages = get_context_for_ai(st.session_state.messages)
        history_obj = []
        for msg in context_messages:
            role_gemini = "model" if msg["role"] == "assistant" else "user"
            history_obj.append({"role": role_gemini, "parts": [msg["content"]]})

        # Salvăm pentru retry în caz de eroare de cheie
        st.session_state["_retry_history"] = history_obj
        st.session_state["_retry_payload"] = [injected]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
            try:
                for text_chunk in run_chat_with_rotation(history_obj, [injected]):
                    full_response += text_chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.empty()
                render_message_with_svg(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message_with_limits(st.session_state.session_id, "assistant", full_response)
                st.session_state.pop("_retry_history", None)
                st.session_state.pop("_retry_payload", None)
            except Exception as e:
                message_placeholder.empty()
                _err_str = str(e)
                _is_rate_limit = any(x in _err_str for x in ["429", "rate_limit", "rate limit", "too many requests", "limită", "Limită"])
                _is_key_err = any(x in _err_str for x in ["epuizat", "invalide", "quota", "API key", "invalid_api_key", "authentication", "401"])
                if _is_rate_limit:
                    st.warning("⏳ Groq este puțin aglomerat — mai încearcă o dată în câteva secunde.", icon="🔄")
                    if st.button("🔄 Reîncercați", key="_retry_quick_action", type="primary"):
                        st.session_state["_pending_retry"] = True
                        st.rerun()
                elif _is_key_err:
                    st.warning("⚠️ Cheia API Groq este invalidă sau quota zilnică s-a epuizat. Verifică cheia în console.groq.com.", icon="🔑")
                    if st.button("🔄 Reîncercați", key="_retry_quick_action", type="primary"):
                        st.session_state["_pending_retry"] = True
                        st.rerun()
                else:
                    st.error(f"❌ Eroare: {e}")
    st.stop()

# ── Handler mesaj în așteptare — materie nedetectată în mod Automat ──
if st.session_state.get("_pending_user_msg") and st.session_state.get("materie_selectata") is None:
    _pending_msg = st.session_state["_pending_user_msg"]

    with st.chat_message("assistant"):
        st.markdown("**La ce materie se referă întrebarea ta?** Alege una din opțiunile de mai jos:")
        # Butoane pentru fiecare materie (fără Automat)
        _materii_optiuni = [(k, v) for k, v in MATERII.items() if v is not None]
        _cols = st.columns(3)
        for i, (label, cod) in enumerate(_materii_optiuni):
            with _cols[i % 3]:
                if st.button(label, key=f"_pick_materie_{cod}", use_container_width=True):
                    # Setăm materia și trimitem mesajul original
                    update_system_prompt_for_subject(cod)
                    st.session_state["_detected_subject"] = cod
                    st.session_state.pop("_pending_user_msg", None)
                    st.session_state["_suggested_question"] = _pending_msg
                    st.rerun()

    st.stop()

# ── Handler întrebare sugerată — ÎNAINTE de afișarea butoanelor ──
if st.session_state.get("_suggested_question"):
    user_input = st.session_state.pop("_suggested_question")
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message_with_limits(st.session_state.session_id, "user", user_input)

    # ── Detecție și routing materie ──
    _materie_manuala = st.session_state.get("materie_selectata")
    _mod_automat = (_materie_manuala is None)

    if not _mod_automat:
        if st.session_state.get("_detected_subject") != _materie_manuala:
            update_system_prompt_for_subject(_materie_manuala)
    else:
        _detected = detect_subject_from_text(user_input)
        _prev_detected = st.session_state.get("_detected_subject")
        if _detected and _detected != _prev_detected:
            update_system_prompt_for_subject(_detected)
            _det_label = _MATERII_LABEL.get(_detected, _detected.capitalize())
            st.toast(f"📚 {_det_label}", icon="🎯")
        elif not _detected and not _prev_detected:
            st.session_state["_pending_user_msg"] = user_input
            st.rerun()

    context_messages = get_context_for_ai(st.session_state.messages)
    history_obj = []
    for msg in context_messages:
        role_gemini = "model" if msg["role"] == "assistant" else "user"
        history_obj.append({"role": role_gemini, "parts": [msg["content"]]})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
        try:
            for text_chunk in run_chat_with_rotation(history_obj, [user_input]):
                full_response += text_chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.empty()
            render_message_with_svg(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_with_limits(st.session_state.session_id, "assistant", full_response)
        except Exception as e:
            st.error(f"❌ Eroare: {e}")
    st.rerun()

# ── Întrebări sugerate per materie — afișate doar când chat-ul e gol ──
# Pool mare de întrebări — 4 alese aleator la fiecare sesiune nouă
INTREBARI_POOL = {
    None: [
        "Explică-mi cum se rezolvă ecuațiile de gradul 2",
        "Ce este fotosinteza și cum funcționează?",
        "Cum se scrie un eseu la BAC?",
        "Explică legea lui Ohm cu un exemplu",
        "Care sunt curentele literare studiate la BAC?",
        "Cum calculez probabilitatea unui eveniment?",
        "Explică-mi structura atomului",
        "Ce este derivata și la ce folosește?",
        "Cum rezolv o problemă cu mișcare uniformă?",
        "Explică-mi reacțiile chimice de bază",
        "Care sunt figurile de stil principale?",
        "Cum funcționează circuitul electric serie vs paralel?",
    ],
    "matematică": [
        "Cum rezolv o ecuație de gradul 2?",
        "Explică-mi derivatele — ce sunt și cum se calculează",
        "Cum calculez aria și volumul unui corp geometric?",
        "Ce este limita unui șir și cum o calculez?",
        "Cum rezolv un sistem de ecuații?",
        "Explică-mi funcțiile monotone și extreme",
        "Ce este matricea și cum fac operații cu ea?",
        "Cum calculez probabilități cu combinări?",
        "Explică-mi trigonometria — formule esențiale",
        "Cum rezolv inecuații de gradul 2?",
        "Ce sunt vectorii și cum fac operații cu ei?",
        "Explică-mi integralele — ce sunt și cum se calculează",
    ],
    "fizică": [
        # Clasa IX — Mecanică
        "Explică legile lui Newton cu exemple concrete",
        "Cum rezolv o problemă cu plan înclinat?",
        "Cum calculez energia cinetică și potențială?",
        "Explică mișcarea uniform accelerată — formule și grafice",
        "Ce este impulsul și cum aplic teorema impulsului?",
        "Cum calculez lucrul mecanic și puterea?",
        "Explică legea lui Arhimede — condiția de plutire",
        "Cum aplic teorema lui Bernoulli în probleme?",
        "Explică mișcarea circulară uniformă — formule",
        "Ce sunt legile lui Kepler și vitezele cosmice?",
        # Clasa X — Termodinamică + Electricitate
        "Ce este legea lui Ohm și cum aplic în circuit?",
        "Cum rezolv o problemă cu circuite mixte (serie+paralel)?",
        "Explică transformările gazelor ideale (izoterm, izobar, izocor)",
        "Cum calculez randamentul unui motor termic?",
        "Ce este curentul alternativ — valori eficace, impedanță?",
        "Explică transformatorul — cum funcționează?",
        "Cum aplic legile lui Kirchhoff într-un circuit?",
        # Clasa XI — Oscilații, unde, optică
        "Explică oscilațiile armonice — pendul și resort",
        "Ce este rezonanța și când apare?",
        "Cum calculez lungimea de undă și viteza unei unde?",
        "Explică interferența undelor — Young",
        "Ce este difracția și cum aplic formula rețelei?",
        "Explică spectrul electromagnetic — tipuri și aplicații",
        "Ce este polarizarea luminii — legea Malus?",
        # Clasa XII — Fizică modernă
        "Explică dilatarea timpului în relativitatea restrânsă",
        "Ce este efectul fotoelectric — ecuația lui Einstein?",
        "Explică modelul Bohr al atomului de hidrogen",
        "Cum calculez energia de legătură a unui nucleu?",
        "Explică dezintegrarea α, β, γ — legi de conservare",
        "Ce este fisiunea nucleară și cum funcționează reactorul?",
        "Explică ipoteza de Broglie — dualism undă-corpuscul",
        "Ce sunt semiconductorii N și P — joncțiunea PN?",
    ],
    "chimie": [
        # Clasa IX — Anorganică & baze fizico-chimice
        "Explică structura atomului și configurația electronică",
        "Cum determin tipul de legătură chimică (ionică, covalentă)?",
        "Ce este echilibrul chimic și principiul Le Châtelier?",
        "Cum calculez pH-ul unui acid/bază tare?",
        "Explică reacțiile redox — oxidare, reducere, bilanț electronic",
        "Cum funcționează pila Daniell — anod, catod, tensiune?",
        "Cum calculez concentrația molară și fac diluții?",
        "Explică coroziunea fierului și metodele de protecție",
        # Clasa X — Organică introductivă
        "Explică-mi alcanii — structură, denumire, reacții",
        "Ce este regula lui Markovnikov — cum o aplic la alchene?",
        "Cum calculez gradul de nesaturare Ω?",
        "Explică izomeria structurală — de catenă, poziție, funcțiune",
        "Cum echilibrez o ecuație chimică pas cu pas?",
        "Cum fac calcule stoechiometrice — cei 5 pași?",
        "Explică reacțiile de esterificare și saponificare",
        "De ce alcoolii au punct de fierbere ridicat? (legături H)",
        "Cum funcționează săpunul — mecanismul spălării?",
        # Clasa XI-XII — Organică avansată & biochimie
        "Explică substituția nucleofilă SN la derivații halogenați",
        "Cum deosebesc aldehidele de cetone? (Tollens, Fehling)",
        "Explică polimerizarea și policondensarea — PVC, nylon",
        "Ce sunt aminoacizii — comportament amfoter, legătură peptidică?",
        "Explică structura glucozei și fructozei — test Fehling",
        "Care este diferența dintre amidon și celuloză?",
        "Explică structura ADN — baze azotate, legături de hidrogen",
        "Ce sunt trigliceridele și cum se face saponificarea grăsimilor?",
        "Explică impactul freonilor asupra stratului de ozon",
    ],
    "limba și literatura română": [
        "Cum structurez un eseu de BAC la Română?",
        "Explică-mi curentele literare principale",
        "Cum analizez o poezie — figuri de stil, prozodie",
        "Care sunt operele obligatorii la BAC Română?",
        "Explică-mi romanul Ion de Rebreanu",
        "Cum caracterizez un personaj literar?",
        "Ce figuri de stil sunt la Eminescu în Luceafărul?",
        "Cum scriu comentariul unui text narativ?",
        "Explică-mi analiza morfologică și sintactică",
        "Care sunt trăsăturile romantismului românesc?",
        "Cum analizez Enigma Otiliei de Călinescu?",
        "Ce este modernismul în literatura română?",
    ],
    "biologie": [
        "Explică-mi mitoza vs meioza",
        "Cum funcționează fotosinteza și respirația celulară?",
        "Ce este ADN-ul și cum funcționează codul genetic?",
        "Explică-mi legile lui Mendel cu pătrat Punnett",
        "Care sunt organitele celulei și funcțiile lor?",
        "Cum funcționează sistemul nervos?",
        "Explică-mi sistemul circulator — inimă și sânge",
        "Ce este fotosinteza — faza luminoasă și Calvin?",
        "Cum funcționează sistemul digestiv?",
        "Explică determinismul sexului și bolile genetice",
        "Ce este ecosistemul și lanțul trofic?",
        "Cum funcționează sistemul endocrin?",
    ],
    "informatică": [
        # Clasa IX - Python baze
        "Explică sortarea prin selecție în Python pas cu pas",
        "Cum funcționează algoritmul lui Euclid pentru cmmdc?",
        "Ce sunt listele în Python? Metode: append, pop, sort",
        "Cum implementez o stivă și o coadă în Python?",
        "Ce este recursivitatea? Exemplu cu factorial în Python",
        "Cum citesc și scriu fișiere text în Python?",
        "Explică-mi funcțiile în Python — parametri și return",
        "Cum fac o interfață grafică simplă cu Tkinter?",
        # Clasa X - colecții + algoritmi
        "Cum funcționează dicționarele în Python? (dict)",
        "Explică diferența dintre set, list, tuple și dict",
        "Cum funcționează căutarea binară? Cod Python",
        "Explică Merge Sort — Divide et Impera în Python",
        "Cum implementez cifrul Cezar în Python?",
        "Ce este QuickSort și cum funcționează?",
        "Cum lucrez cu matrici (tablouri 2D) în Python și C++?",
        "Explică-mi struct în C++ cu exemple",
        # Clasa XI - grafuri, arbori, algoritmi avansați
        "Ce sunt grafurile? Explică BFS și DFS cu exemple",
        "Cum funcționează algoritmul Dijkstra?",
        "Explică backtracking-ul cu problema N-Reginelor",
        "Ce este programarea dinamică? Exemplu cu rucsacul",
        "Cum implementez un arbore binar de căutare?",
        "Explică algoritmii Prim și Kruskal pentru MST",
        "Ce sunt listele înlănțuite și cum le implementez?",
        "Roy-Floyd — drumuri minime între toate perechile",
        # Clasa XII - BD, SQL, ML
        "Explică-mi modelul entitate-relație (ERD)",
        "SQL: cum fac un JOIN între două tabele?",
        "Ce este normalizarea bazelor de date? FN1, FN2, FN3",
        "Cum conectez Python la o bază de date SQLite?",
        "Introduc-ți Pandas — DataFrame și operații de bază",
        "Cum antrenez un model KNN cu scikit-learn?",
        "Ce este K-Means și cum funcționează clustering-ul?",
        "Explică regresia liniară cu un exemplu în Python",
    ],
    "geografie": [
        "Care sunt unitățile de relief ale României?",
        "Explică-mi clima României — regiuni și factori",
        "Care sunt râurile principale din România?",
        "Explică formarea Munților Carpați",
        "Care sunt vecinii României și granițele?",
        "Explică-mi Delta Dunării — caracteristici",
        "Care sunt resursele naturale ale României?",
        "Explică populația și orașele mari din România",
        "Ce sunt continentele — caracteristici principale?",
        "Explică-mi coordonatele geografice",
        "Care sunt problemele de mediu din România?",
        "Explică clima Europei — zone climatice",
    ],
    "istorie": [
        "Explică Marea Unire din 1918 — cauze și consecințe",
        "Care au fost reformele lui Alexandru Ioan Cuza?",
        "Explică-mi perioada comunistă în România",
        "Ce s-a întâmplat la Revoluția din 1989?",
        "Cine a fost Ștefan cel Mare și care sunt realizările lui?",
        "Explică Primul Război Mondial — România",
        "Ce a fost Revoluția de la 1848 în Țările Române?",
        "Explică domnia lui Mihai Viteazul și prima unire",
        "Care au fost cauzele Independenței din 1877?",
        "Explică perioada interbelică în România",
        "Ce a fost Holocaustul și implicarea României?",
        "Cine a fost Carol I și ce a realizat?",
    ],
    "limba franceză": [
        "Explică-mi Passé Composé vs Imparfait",
        "Cum se acordă participiul trecut cu avoir și être?",
        "Explică Subjonctivul — când și cum se folosește",
        "Cum structurez un eseu în franceză?",
        "Explică-mi Futur Simple vs Futur Proche",
        "Cum funcționează pronumele relative (qui, que, dont)?",
        "Explică condiționalul prezent și trecut",
        "Ce sunt verbele neregulate esențiale în franceză?",
        "Cum exprim cauza și consecința în franceză?",
        "Explică-mi acordul adjectivelor în franceză",
    ],
    "limba engleză": [
        "Explică Present Perfect vs Past Simple",
        "Cum funcționează propozițiile condiționale (tip 1, 2, 3)?",
        "Explică vocea pasivă în engleză",
        "Cum scriu un eseu argumentativ în engleză?",
        "Explică reported speech — vorbire indirectă",
        "Ce sunt modal verbs și când le folosesc?",
        "Cum funcționează articolele a/an/the în engleză?",
        "Explică-mi timpurile verbale — ghid complet",
        "Cum scriu o scrisoare formală în engleză?",
        "Explică relative clauses (who, which, that)",
    ],
    "limba germană": [
        # Clasa IX — A1/A2, baze
        "Explică genul substantivelor în germană — der, die, das",
        "Cum conjugăm verbele la prezent (Präsens) în germană?",
        "Explică ordinea cuvintelor în propoziția germană (Satzstellung)",
        "Ce sunt verbele modale în germană? können, müssen, dürfen...",
        "Cum formăm întrebările în germană — W-Fragen și Da/Nein-Fragen?",
        "Explică-mi cazurile în germană — nominativ, acuzativ, dativ",
        "Cum funcționează verbele separabile (trennbare Verben)?",
        # Clasa X — A2/B1
        "Explică Perfekt vs Präteritum — când folosesc fiecare?",
        "Cum se formează Partizip II pentru Perfekt?",
        "Care verbe cer sein și care haben la Perfekt?",
        "Cum compar adjectivele în germană? gut → besser → am besten",
        "Explică propoziția subordonată cu weil, dass, obwohl — verbul la sfârșit!",
        "Ce sunt verbele reflexive în germană? (sich waschen, sich freuen)",
        # Clasa XI — B1
        "Explică Konjunktiv II — würde, wäre, hätte și când îl folosesc",
        "Cum funcționează pronumele relative în germană?",
        "Explică Passiv în germană — werden + Partizip II",
        "Cum scriu un CV și o scrisoare de intenție în germană?",
        "Explică Plusquamperfekt — mai-mult-ca-perfectul în germană",
        # Clasa XII — B1/B2
        "Explică Konjunktiv I pentru vorbire indirectă (Indirekte Rede)",
        "Ce sunt conectorii dubli în germană? entweder…oder, sowohl…als auch",
        "Cum structurez un eseu argumentativ în germană (BAC)?",
        "Explică Infinitivkonstruktionen cu zu în germană",
        "Cum folosesc genitivul în texte formale germane?",
    ],
}

if not st.session_state.get("messages") and not st.session_state.get("pedagogie_mode"):
    materie_curenta = st.session_state.get("materie_selectata")

    if materie_curenta is None:
        # Mod Automat — afișăm selector de materie pe pagina principală
        st.markdown("##### 📚 Selectează materia")
        _materii_butoane = [(k, v) for k, v in MATERII.items() if v is not None]
        _cols = st.columns(2)
        for i, (label, cod) in enumerate(_materii_butoane):
            with _cols[i % 2]:
                if st.button(label, key=f"pick_mat_{cod}", use_container_width=True):
                    # Setăm materia în selector și în session_state
                    st.session_state.materie_selectata = cod
                    st.session_state["_detected_subject"] = cod
                    st.session_state["system_prompt"] = get_system_prompt(
                        materie=cod,
                        pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                        mod_avansat=st.session_state.get("mod_avansat", False),
                        mod_strategie=st.session_state.get("mod_strategie", False),
                        mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
                    )
                    st.rerun()
    else:
        # Materie selectată — afișăm întrebări sugerate pentru materia respectivă
        pool = INTREBARI_POOL.get(materie_curenta, INTREBARI_POOL[None])
        _sugg_key = f"_sugg_list_{st.session_state.session_id}"
        _sugg_materie_key = f"_sugg_materie_{st.session_state.session_id}"
        if (
            _sugg_key not in st.session_state
            or st.session_state.get(_sugg_materie_key) != materie_curenta
        ):
            st.session_state[_sugg_key] = random.sample(pool, min(4, len(pool)))
            st.session_state[_sugg_materie_key] = materie_curenta
        intrebari = st.session_state[_sugg_key]

        col_title, col_refresh = st.columns([4, 1])
        with col_title:
            st.markdown("##### 💡 Cu ce începem azi?")
        with col_refresh:
            if st.button("🔄", key="_refresh_sugg_btn", help="Alte întrebări"):
                st.session_state.pop(_sugg_key, None)
                st.rerun()
        cols = st.columns(2)
        for i, intrebare in enumerate(intrebari):
            with cols[i % 2]:
                if st.button(intrebare, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["_suggested_question"] = intrebare
                    st.rerun()

# === AVERTISMENT OFFLINE ===
if st.session_state.get("_history_may_be_incomplete"):
    st.warning(
        "📴 **Mod offline** — istoricul afișat poate fi incomplet față de baza de date. "
        "Reconectarea se face automat când rețeaua revine.",
        icon="⚠️"
    )
    if st.button("🔄 Verifică conexiunea acum", key="_check_conn_btn"):
        # Forțăm re-marcarea ca online pentru a testa
        st.session_state.pop("_sb_online", None)
        st.session_state.pop("_history_may_be_incomplete", None)
        st.rerun()

# === HANDLER RETRY după eroare de cheie API ===
# Dacă utilizatorul a apăsat "Reîncercați" după o eroare de cheie, reluăm cererea
# cu aceleași history + payload salvate ÎNAINTE de eroare.
if st.session_state.pop("_pending_retry", False):
    _retry_history  = st.session_state.get("_retry_history")
    _retry_payload  = st.session_state.get("_retry_payload")
    if _retry_history is not None and _retry_payload is not None:
        with st.chat_message("assistant"):
            _rph = st.empty()
            _rph.markdown(TYPING_HTML, unsafe_allow_html=True)
            _rfull = ""
            try:
                for _chunk in run_chat_with_rotation(_retry_history, _retry_payload):
                    _rfull += _chunk
                    if "<svg" in _rfull or ("<path" in _rfull and "stroke=" in _rfull):
                        _rph.markdown(_rfull.split("<path")[0] + "\n\n*🎨 Domnul Profesor desenează...*\n\n▌")
                    else:
                        _rph.markdown(_rfull + "▌")
                _rph.empty()
                render_message_with_svg(_rfull)
                st.session_state.messages.append({"role": "assistant", "content": _rfull})
                save_message_with_limits(st.session_state.session_id, "assistant", _rfull)
                st.session_state.pop("_retry_history", None)
                st.session_state.pop("_retry_payload", None)
            except Exception as _re:
                _rph.empty()
                st.error(f"❌ Eroare și la reîncercare: {_re}")
    st.stop()

# === CHAT INPUT ===
if user_input := st.chat_input("Întreabă profesorul..."):

    # --- Debounce: blochează mesaje duplicate trimise rapid ---
    now_ts = time.time()
    last_msg = st.session_state.get("_last_user_msg", "")
    last_ts  = st.session_state.get("_last_msg_ts", 0)
    DEBOUNCE_SECONDS = 2.5

    if user_input.strip() == last_msg.strip() and (now_ts - last_ts) < DEBOUNCE_SECONDS:
        st.toast("⏳ Mesaj duplicat ignorat.", icon="🔁")
        st.stop()

    st.session_state["_last_user_msg"] = user_input
    st.session_state["_last_msg_ts"]  = now_ts

    # FIX BUG 1: Afișează și salvează mesajul utilizatorului ÎNAINTE de răspunsul AI
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message_with_limits(st.session_state.session_id, "user", user_input)

    # ── Detecție și routing materie ──
    _materie_manuala = st.session_state.get("materie_selectata")  # None = mod Automat
    _mod_automat = (_materie_manuala is None)

    if not _mod_automat:
        # Mod manual: selectorul are prioritate — asigurăm prompt-ul corect
        if st.session_state.get("_detected_subject") != _materie_manuala:
            update_system_prompt_for_subject(_materie_manuala)
        # FIX Bug 3: avertizăm dacă textul pare să fie pentru altă materie
        # (detectăm din mesaj, comparăm cu selecția manuală — toast non-blocant)
        _detected_in_msg = detect_subject_from_text(user_input)
        if (
            _detected_in_msg
            and _detected_in_msg != _materie_manuala
            and _detected_in_msg != "pedagogie"
            and not st.session_state.get(f"_mismatch_warned_{st.session_state.session_id}")
        ):
            _sel_label = _MATERII_LABEL.get(_materie_manuala, _materie_manuala or "materia selectată")
            _det_label = _MATERII_LABEL.get(_detected_in_msg, _detected_in_msg.capitalize())
            st.toast(
                f"💡 Mesajul pare să fie despre {_det_label}, dar ești pe {_sel_label}. "
                f"Schimbă materia din sidebar dacă vrei răspuns specializat.",
                icon="⚠️"
            )
            st.session_state[f"_mismatch_warned_{st.session_state.session_id}"] = True

    else:
        # Mod automat: încearcă să detecteze materia din mesaj
        _detected = detect_subject_from_text(user_input)
        _prev_detected = st.session_state.get("_detected_subject")

        if _detected:
            # Detectat cu succes
            if _detected != _prev_detected:
                update_system_prompt_for_subject(_detected)
                _det_label = _MATERII_LABEL.get(_detected, _detected.capitalize())
                st.toast(f"📚 {_det_label}", icon="🎯")
            # FIX: resetăm flag-ul de mismatch la schimbare de materie
            for _k in [k for k in st.session_state.keys() if k.startswith("_mismatch_warned_")]:
                del st.session_state[_k]
        else:
            # Nu s-a putut detecta materia — salvăm mesajul și întrebăm elevul
            st.session_state["_pending_user_msg"] = user_input
            st.rerun()

    context_messages = get_context_for_ai(st.session_state.messages)
    history_obj = []
    for msg in context_messages:
        role_gemini = "model" if msg["role"] == "assistant" else "user"
        history_obj.append({"role": role_gemini, "parts": [msg["content"]]})
    
    final_payload = []
    if media_content:
        _uf = st.session_state.get("_current_uploaded_file_meta", {})
        fname = _uf.get("name", "")
        ftype = _uf.get("type", "") or ""
        if ftype.startswith("image/"):
            import base64 as _b64
            _b64_str = _b64.b64encode(media_content).decode("utf-8")
            # Trimitem imaginea ca URL base64 — Groq vision o procesează direct
            final_payload.append(
                f"data:{ftype};base64,{_b64_str}"
            )
            final_payload.append(
                "Elevul ți-a trimis o imagine. Analizează-o vizual complet: "
                "descrie ce vezi (obiecte, persoane, text, culori, forme, diagrame, exerciții scrise de mână) "
                "și răspunde la întrebarea elevului ținând cont de tot conținutul vizual."
            )
        else:
            final_payload.append(
                f"Elevul ți-a trimis documentul '{fname}'. "
                "Citește și analizează tot conținutul înainte de a răspunde."
            )
    # Detectăm dacă e o cerere de desen — injectăm instrucțiune strictă în payload
    _DESEN_KEYWORDS = ["desen", "desenează", "desenez", "schemă", "schema", "diagramă",
                       "diagrama", "arată-mi", "arata-mi", "ilustrează", "ilustreaza",
                       "fă un desen", "fa un desen", "fă-mi un desen", "fa-mi un desen"]
    _is_desen_request = any(kw in user_input.lower() for kw in _DESEN_KEYWORDS)
    if _is_desen_request:
        final_payload.append(
            "INSTRUCȚIUNE SISTEM: Răspunde EXCLUSIV cu desenul SVG în format "
            "[[DESEN_SVG]]<svg ...>...</svg>[[/DESEN_SVG]]. "
            "NU include text, explicații, rezolvări sau orice alt conținut în afara blocului SVG. "
            "DOAR desenul, nimic altceva. Mesajul elevului: "
        )
    final_payload.append(user_input)

    # Salvăm payload-ul ÎNAINTE de apelul AI — dacă cheia se epuizează în stream,
    # elevul poate reîncerca fără să retrimită mesajul manual.
    st.session_state["_retry_history"] = history_obj
    st.session_state["_retry_payload"] = final_payload

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Typing indicator înainte să înceapă streaming-ul
        message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        try:
            stream_generator = run_chat_with_rotation(history_obj, final_payload)
            first_chunk = True

            for text_chunk in stream_generator:
                full_response += text_chunk
                if first_chunk:
                    first_chunk = False  # typing indicator dispare la primul chunk

                if "<svg" in full_response or ("<path" in full_response and "stroke=" in full_response):
                    message_placeholder.markdown(
                        full_response.split("<path")[0] + "\n\n*🎨 Domnul Profesor desenează...*\n\n▌"
                    )
                else:
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.empty()
            render_message_with_svg(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_with_limits(st.session_state.session_id, "assistant", full_response)
            # Răspuns reușit — curățăm datele de retry
            st.session_state.pop("_retry_history", None)
            st.session_state.pop("_retry_payload", None)

        except Exception as e:
            message_placeholder.empty()
            err_str = str(e)
            _is_rate_limit2 = any(x in err_str for x in ["429", "rate_limit", "rate limit", "too many requests", "limită", "Limită"])
            _is_key_err2 = any(x in err_str for x in ["epuizat", "invalide", "quota", "API key", "invalid_api_key", "authentication", "401"])
            if _is_rate_limit2:
                st.warning("⏳ Groq este puțin aglomerat — mai încearcă o dată în câteva secunde.", icon="🔄")
                if st.button("🔄 Reîncercați răspunsul", key="_retry_after_key_error", type="primary"):
                    st.session_state["_pending_retry"] = True
                    st.rerun()
            elif _is_key_err2:
                st.warning(
                    "⚠️ Cheia API Groq este invalidă sau quota zilnică s-a epuizat. "
                    "Verifică cheia în console.groq.com.",
                    icon="🔑"
                )
                if st.button("🔄 Reîncercați răspunsul", key="_retry_after_key_error", type="primary"):
                    st.session_state["_pending_retry"] = True
                    st.rerun()
            else:
                st.error(f"❌ Eroare: {e}")
