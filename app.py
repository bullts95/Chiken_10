# app.py
import sys
import io
import os

# --- 【重要】Streamlit Cloud用 SQLite対策 ---
# Streamlit CloudのデフォルトSQLiteは古いため、pysqlite3を使って新しいバージョンに置き換えます
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # ローカル環境などpysqlite3がない場合は何もしない
# -------------------------------------------

import pandas as pd
import streamlit as st
from typing import Annotated, TypedDict

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx # ngrokヘッダー用

# RAG用 Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------------------------------------------
# ▼ 設定・定数
# -----------------------------------------------------------------
st.set_page_config(page_title="Legal Analysis Chatbot", layout="wide")
st.title("🤖 親権喪失・停止事例分析チャットボット (Team Demo)")

# サイドバー設定
with st.sidebar:
    st.header("🔐 接続設定")
    
    # 1. チーム認証 (簡易パスワード)
    team_password = st.secrets.get("auth", {}).get("team_password", "demo1234")
    input_password = st.text_input("Team Password", type="password")
    is_authenticated = input_password == team_password

    if is_authenticated:
        st.success("認証OK")
    else:
        st.warning("パスワードを入力してください")

    # 2. ngrok URL入力 (毎回変わるため手動入力)
    st.markdown("---")
    st.write("🔗 **LLM接続 (ngrok)**")
    ngrok_url = st.text_input(
        "ngrok URL", 
        placeholder="https://xxxx.ngrok-free.app",
        disabled=not is_authenticated,
        help="ローカルPCで発行されたngrokのHTTPS URLを入力してください"
    )
    
    # モデル名は固定または入力
    model_name = "local-model"

    st.markdown("---")
    st.write("📂 **データ設定**")
    # CSVはリポジトリ同梱を基本とするが、アップロードも許可
    uploaded_file = st.file_uploader("CSVファイルを上書き (任意)", type=["csv"])
    
    # DBパス: Streamlit Cloudのパス構成に合わせて相対パス指定
    # リポジトリ直下に 'DB' フォルダがある前提
    db_refs_path = st.text_input("参考文献DBパス", value="./DB")

# -----------------------------------------------------------------
# ▼ 1. データのロード
# -----------------------------------------------------------------
@st.cache_data
def load_data(file_or_path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_or_path)
    except Exception as e:
        return pd.DataFrame()

# データ読み込みロジック
df_global = pd.DataFrame()

if uploaded_file is not None:
    df_global = load_data(uploaded_file)
    st.sidebar.success(f"Uploadデータ使用: {len(df_global)}件")
else:
    # リポジトリ内のデフォルトCSVを探す
    default_csv = "cases_list_integrated.csv"
    if os.path.exists(default_csv):
        df_global = load_data(default_csv)
        st.sidebar.info(f"リポジトリ内データ使用: {len(df_global)}件")
    else:
        st.sidebar.warning("CSVが見つかりません。")

# -----------------------------------------------------------------
# ▼ 2. ツール定義
# -----------------------------------------------------------------
# LLMクライアント初期化用のヘルパー関数 (ngrokヘッダー対応)
def get_llm_client(base_url, api_key="lm-studio"):
    custom_client = httpx.Client(headers={"ngrok-skip-browser-warning": "true"})
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.0,
        http_client=custom_client
    )

@tool
def get_case_details(case_id: str) -> str:
    """単一の case_id を受け取り、CSVからその事案の詳細を取得します。"""
    if df_global.empty: return "データがロードされていません。"
    try:
        # IDの型揺らぎ吸収
        try:
            target_id = int(case_id.strip())
            record = df_global[df_global['case_id'] == target_id]
        except:
            target_id = case_id.strip()
            record = df_global[df_global['case_id'].astype(str) == target_id]

        if record.empty:
            return f"case_id '{case_id}' は見つかりませんでした。"
            
        row = record.iloc[0]
        details = [f"--- case_id: {row['case_id']} の詳細 ---"]
        for col, val in row.items():
            if pd.notna(val) and str(val).strip() != "":
                details.append(f"  - {col}: {val}")
        return "\n".join(details)
    except Exception as e:
        return f"エラー: {e}"

@tool
def analyze_statistics(query: str) -> str:
    """統計分析専用ツール。Pythonコードを生成・実行して結果を返します。"""
    # ツール内でLLMを呼ぶため、グローバル変数または引数からURL取得が必要
    # ここでは簡易的にst.session_stateなどを参照せず、再初期化で対応
    if not ngrok_url:
        return "エラー: ngrok URLが設定されていません。"
    
    # ngrok経由でCode生成用LLMを呼ぶ
    base_url_v1 = ngrok_url.rstrip("/") + "/v1"
    coder_llm = get_llm_client(base_url_v1)

    columns_list = ", ".join(df_global.columns.tolist()) if not df_global.empty else "なし"
    
    prompt = f"""
    あなたは優秀なPythonデータアナリストです。
    ユーザーの質問に答えるための Pandas コードのみを書いてください。
    変数名は 'df' を使用。列名: [{columns_list}]
    
    質問: {query}
    
    ルール:
    - Pythonコードのみ出力(Markdownタグ不要)
    - 結果は print() で出力
    - グラフ描画禁止
    """

    try:
        response = coder_llm.invoke(prompt)
        code = response.content.replace("```python", "").replace("```", "").strip()
        
        local_env = {'df': df_global.copy(), 'pd': pd}
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            exec(code, {}, local_env)
            result = redirected_output.getvalue()
        except Exception as e:
            result = f"コード実行エラー: {e}"
        finally:
            sys.stdout = old_stdout
            
        return f"【分析結果】\n{result}" if result.strip() else "結果なし"

    except Exception as e:
        return f"分析エラー: {e}"

# -----------------------------------------------------------------
# ▼ 3. RAG機能 (参考文献検索)
# -----------------------------------------------------------------
@st.cache_resource(show_spinner="参考文献DBをロード中...")
def get_ref_retriever(db_path):
    """
    Streamlit Cloud上でHuggingFaceモデルをロードし、Chromaを読み込む。
    注意: 初回はモデルダウンロード(数GB)が走るため時間がかかります。
    """
    try:
        if not os.path.exists(db_path):
            return None
            
        # モデル設定 (CPU動作)
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model_name = "intfloat/multilingual-e5-large"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        db_refs = Chroma(persist_directory=db_path, embedding_function=embeddings)
        return db_refs.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"DBロードエラー: {e}")
        return None

def search_references_action(user_query):
    if not os.path.exists(db_refs_path):
        return "⚠️ リポジトリ内に 'DB' フォルダが見つかりません。"

    retriever = get_ref_retriever(db_refs_path)
    if not retriever:
        return "DBの初期化に失敗しました。"
    
    try:
        docs = retriever.invoke(user_query)
        if not docs: return "関連文献なし"
        
        result_text = f"**Q: {user_query}** に関連する参考文献:\n\n"
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "不明")
            content = doc.page_content.replace("\n", " ")[:300]
            result_text += f"**[{i}] {source}**\n> {content}...\n\n"
        return result_text
    except Exception as e:
        return f"検索エラー: {e}"

# -----------------------------------------------------------------
# ▼ 4. LangGraph 構築
# -----------------------------------------------------------------
@st.cache_resource
def build_graph(_llm_client):
    # ツールリスト
    tools = [get_case_details, analyze_statistics]
    
    # ツールバインド
    llm_with_tools = _llm_client.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def agent_node(state: AgentState):
        messages = state['messages']
        system_prompt = SystemMessage(content="""
        あなたは法律専門家アシスタントです。
        ツールを使用して事実に基づき回答してください。
        """)
        
        # システムプロンプト管理
        if not isinstance(messages[0], SystemMessage):
            messages = [system_prompt] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    
    def should_continue(state):
        last_message = state['messages'][-1]
        if last_message.tool_calls: return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# -----------------------------------------------------------------
# ▼ 5. メイン処理
# -----------------------------------------------------------------

# 認証 & URLチェック
if not is_authenticated or not ngrok_url:
    st.info("👈 サイドバーからパスワードとngrok URLを入力してください。")
    st.stop()

# アプリ初期化
base_url_v1 = ngrok_url.rstrip("/") + "/v1"
main_llm_client = get_llm_client(base_url_v1)
app = build_graph(main_llm_client)

# チャット履歴
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_output"):
            with st.expander("詳細データ"):
                st.text(message["tool_output"])

# 参考文献ボタン
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if len(st.session_state.messages) >= 2:
        last_user_query = st.session_state.messages[-2]["content"]
        if st.button("📚 参考文献も検索する"):
            with st.spinner("検索中... (初回はモデルロードに時間がかかります)"):
                ref_result = search_references_action(last_user_query)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ref_result,
                    "tool_output": None
                })
                st.rerun()

# チャット入力
if prompt := st.chat_input("質問を入力..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if df_global.empty:
        st.error("CSVがありません。")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("思考中...", expanded=True)
        
        # LangGraph実行用メッセージ変換
        lc_messages = []
        for m in st.session_state.messages:
            role = "user" if m["role"] == "user" else "assistant"
            lc_messages.append(HumanMessage(content=m["content"]) if role == "user" else AIMessage(content=m["content"]))

        try:
            inputs = {"messages": lc_messages}
            full_response = ""
            captured_outputs = []

            for event in app.stream(inputs, stream_mode="values"):
                last_msg = event["messages"][-1]
                
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        status_container.write(f"🛠️ {tc['name']}")
                
                elif last_msg.type == "tool":
                    captured_outputs.append(last_msg.content)
                
                elif isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    full_response = last_msg.content
                    message_placeholder.markdown(full_response)

            status_container.update(label="完了", state="complete", expanded=False)
            
            if full_response:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "tool_output": "\n".join(captured_outputs) if captured_outputs else None
                })
                st.rerun()
                
        except Exception as e:
            status_container.update(label="エラー", state="error")
            st.error(f"通信エラー: {e}")
            st.info("ngrokのURLが正しいか確認してください。")