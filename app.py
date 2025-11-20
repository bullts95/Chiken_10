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
st.title("🤖 親権喪失・停止事例表分析チャットボット (Team Demo)")

st.markdown("""
事例表の検索をしてくれます。事例番号は、1桁のcase_idに変換しました。1桁目が1のものが親権喪失、2のものが親権停止で、残り3桁は、もともとの事例番号からハイフンを除いたものです。
例：親権喪失の3-1->1031、親権停止の20->2200
""")

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
    default_csv = "Cases_list_integrated.csv"
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
    """
    【重要】統計分析だけでなく、「条件に合う事案の検索」にも使用します。
    ユーザーが具体的な case_id を指定せず、「〜な事案について教えて」「検察官の申立ては？」のように質問した場合、
    必ずこのツールを使って Pandas コードを実行し、該当するデータの抽出や要約を行ってください。
    """
    # ... (以下の中身のコードは変更なし) ...
    # ngrok経由でCode生成用LLMを呼ぶ
    if not ngrok_url:
        return "エラー: ngrok URLが設定されていません。"
    
    base_url_v1 = ngrok_url.rstrip("/") + "/v1"
    coder_llm = get_llm_client(base_url_v1)

    columns_list = ", ".join(df_global.columns.tolist()) if not df_global.empty else "なし"
    
    # ---------------------------------------------------------
    # ▼ 列名の定義辞書 (ご提示のCSV定義に合わせて作成)
    # ---------------------------------------------------------
    column_mapping_prompt = """
    【重要: 列名の定義辞書 (日本語の質問 -> 変数名)】
    ユーザーの質問に含まれる単語は、以下のルールに従って列名に変換してください。

    1. 事案の基本情報
       - 「事例ID」「ID」 -> 'case_id'
       - 「裁判所」 -> 'court_type'
       - 「審判日」「日付」 -> 'decision_date' (形式: YYYY-MM-DD)
       - 「申立内容」「申立ての種類」 -> 'petition_type'
          ※「親権停止」,「親権喪失」,「親権停止取消し」のいずれかで、他に入るものはない。勝手に作らないこと
       - 「虐待」「虐待の類型」 -> 'abuse_type_1', 'abuse_type_2, ...'


    2. 当事者 (タグ情報)
       - 「申立人」 -> 'petitioner_A_tag', 'petitioner_B_tag' (人数は 'petitioner_count')
       - 「事件本人」「相手方」「親」 -> 'subject_A_tag', 'subject_B_tag' (人数は 'subject_count')
          ※ [父], [母], [検察官], [児童相談所長] などが含まれる。「未成年者本人」とは、[子]のことを指す。

    3. 子どもの情報 (重要: 最大4人まで列が横に展開されています)
       - 「子の代理人の有無」 -> 'child_counsel'
          ※この列がTrueでなければ、「子」又は「未成年者」に代理人はついていなかったことになる。
       - 「子の人数」 -> 'child_count'
       - 「子」「子供」 -> 'child_A_tag', 'child_B_tag', 'child_C_tag', 'child_D_tag'
       - 「年齢」 -> 'child_A_age', 'child_B_age', 'child_C_age', 'child_D_age'
       - 「監護状況」 -> 'child_A_custody', 'child_B_custody', 'child_C_custody', 'child_D_custody'


    4. 審判結果 (最も重要)
       - 「結果」「審判」 -> 'child_A_result', 'child_B_result', 'child_C_result', 'child_D_result'      
       - 「停止期間」「月数」 -> 'child_A_suspension_months', ... (数値)
       - 「停止終了日」 -> 'child_A_suspension_end_date', ...

    【コード生成時の特別ルール】
    ルール1 (複数列の検索):
      「結果が親権喪失の事案」のように検索する場合、対象は子供ごとに分かれています。
      必ず `child_A_result`, `child_B_result`, `child_C_result`, `child_D_result`のいずれかが条件を満たすか確認してください。
      (例: `df[(df['child_A_result'] == '親権喪失') | (df['child_B_result'] == '親権喪失')]`)

    ルール2 (特定の子の指定):
      「第1子」や「長男・長女」等の指定がない限り、基本的には 'child_A_...' (第1子) を主として分析してください。
      ただし「全体」や「件数」を聞かれた場合は、事案単位(`case_id`)でカウントしてください。
    """

    # 4. プロンプトの構築 (以前と同じ構成に、上記の辞書を埋め込み)
    prompt = f"""
    あなたは優秀なPythonデータアナリストです。
    ユーザーの質問に答えるための Pandas コードのみを書いてください。
    
    【データセット情報】
    変数名: df
    全ての列名: {columns_list}
    
    {column_mapping_prompt}
    
    【ユーザーの質問】
    {query}
    
    【出力ルール】
    - 上記の辞書にある列名を正確に使用すること。存在しない列名(例: 'judgment', 'result_all')は禁止。
    - Pythonコードのみを出力 (Markdownタグなし)。
    - 結果は必ず `print()` で出力。
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

def search_references_action(user_query, llm_client):
    if not os.path.exists(db_refs_path):
        return "⚠️ リポジトリ内に 'DB' フォルダが見つかりません。"

    retriever = get_ref_retriever(db_refs_path)
    if not retriever:
        return "DBの初期化に失敗しました。"
    
    try:
        docs = retriever.invoke(user_query)
        if not docs: return "関連文献なし"
        
        # 1. コンテキストの構築
        context_list = []
        ref_display_text = ""
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "不明")
            content = doc.page_content.replace("\n", " ")
            
            # LLM入力用
            context_list.append(f"文献[{i}] (出典: {source}):\n{content}")
            
            # 表示用 (抜粋)
            ref_display_text += f"**[{i}] {source}**\n> {content[:300]}...\n\n"

        context_str = "\n\n".join(context_list)

        # 2. LLMによる回答生成
        prompt = f"""
        あなたは法律の専門家です。以下の【参考文献】の内容のみに基づいて、ユーザーの【質問】に回答してください。
        回答の際は、どの文献を参照したか（例: [1]）を文中に明記してください。
        参考文献に答えが含まれていない場合は、「参考文献には記載がありません」と答えてください。

        【参考文献】
        {context_str}

        【質問】
        {user_query}
        """
        
        response = llm_client.invoke(prompt)
        answer = response.content

        # 3. 結果の結合
        final_output = f"### 🤖 参考文献に基づく回答\n{answer}\n\n---\n### 📚 参照された文献\n{ref_display_text}"
        return final_output

    except Exception as e:
        return f"検索・生成エラー: {e}"

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
        # システムプロンプトを強化
        system_prompt = SystemMessage(content="""
        あなたは法律専門家のアシスタントです。
        以下のルールに従って回答してください。

        1. **具体的な case_id がない質問の場合:**
           - 即座に「IDがない」と断らず、必ず `analyze_statistics` ツールを使ってデータを検索してください。
           - 例: "df[df['petitioner'] == '検察官']" のようなコードを実行して事例を探します。

        2. **データの読み方:**
           - `petition_type` は「申立内容」、`child_..._result` は「最終判断」です。
           - `result: 却下` は「申立てが認められなかった」という意味です。
        
        ユーザーの役に立つよう、データに基づいた具体的な回答を心がけてください。
        """)
        
        # システムプロンプト管理
        if not isinstance(messages[0], SystemMessage):
            messages = [system_prompt] + messages
        else:
            messages[0] = system_prompt # 既存があれば上書き
        
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
                ref_result = search_references_action(last_user_query, main_llm_client)
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