import streamlit as st
from openai import OpenAI, DefaultHttpxClient

st.set_page_config(page_title="Team Demo Chat", layout="centered")

# --- サイドバー設定（接続情報入力） ---
with st.sidebar:
    st.header("🔌 接続設定")
    
    # パスワード確認
    entered_password = st.text_input("Team Password", type="password")
    # secretsが未設定の場合のエラーハンドリングを追加
    team_password = st.secrets.get("auth", {}).get("team_password", "demo")
    is_authenticated = entered_password == team_password
    
    if not is_authenticated:
        st.warning("パスワードを入力してください")
    else:
        st.success("認証OK")
    
    # ngrok URLの入力
    ngrok_url = st.text_input(
        "ngrok URL (https://...)", 
        placeholder="https://xxxx-xxxx.ngrok-free.app",
        disabled=not is_authenticated
    )
    
    if st.button("接続テスト & リセット", disabled=not is_authenticated):
        st.session_state.messages = [] 
        st.rerun()

# --- メインチャット画面 ---
st.title("🤖 技術デモ用チャット")

if not is_authenticated or not ngrok_url:
    st.info("👈 左のサイドバーからパスワードとngrokのURLを入力してください。")
    st.stop()

# OpenAIクライアントの初期化（ngrok警告回避ヘッダー付き）
try:
    base_url = ngrok_url.rstrip("/") + "/v1"
    
    client = OpenAI(
        base_url=base_url,
        api_key="lm-studio",
        http_client=DefaultHttpxClient(
            headers={"ngrok-skip-browser-warning": "true"}
        )
    )
except Exception as e:
    st.error(f"クライアント初期化エラー: {e}")
    st.stop()

# チャット履歴
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 入力と応答
if prompt := st.chat_input("メッセージを入力..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # モデル名はLM Studio側でロードされているものが使われるため適当な文字列でOK
            stream = client.chat.completions.create(
                model="local-model", 
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            response_placeholder.markdown("🚨 **Connection Error**")
            st.error(f"エラー詳細: {e}")
            st.info("💡 ngrokのURLが正しいか、PCでLM StudioのサーバーがONになっているか確認してください。")