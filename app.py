import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from unidecode import unidecode
import re
from io import BytesIO
import base64
from datetime import datetime
import community.community_louvain as community_louvain

# ==================== CHUẨN HÓA DỮ LIỆU ====================

def normalize_name(name: str) -> str:
    """Chuẩn hóa tên người: bỏ dấu, viết hoa, gộp khoảng trắng"""
    if pd.isna(name):
        return ''
    name = str(name).strip()
    name = unidecode(name)
    name = re.sub(r'\s+', ' ', name)
    name = name.upper()
    return name

def normalize_account(acc: str) -> str:
    """Chuẩn hóa mã tài khoản: bỏ khoảng trắng, viết hoa"""
    if pd.isna(acc):
        return ''
    acc = re.sub(r'\s+', '', str(acc).strip()).upper()
    return acc

def parse_amount(x):
    """Chuyển đổi số tiền từ dạng string có dấu chấm/phẩy"""
    if pd.isna(x):
        return 0
    s = str(x).replace('.', '').replace(',', '')
    s = re.sub(r'[^\d\-]', '', s)
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return 0

# ==================== XỬ LÝ DỮ LIỆU ====================

def load_and_process_data(flow_file, group_file=None, exclude_internal=True, transaction_type='Cả hai'):
    """Đọc và xử lý dữ liệu từ 2 file"""
    
    # Đọc file luồng tiền
    if flow_file.name.endswith('.csv'):
        df_flow = pd.read_csv(flow_file)
    else:
        df_flow = pd.read_excel(flow_file)
    
    # Chuẩn hóa cột
    df_flow['Nguoi_nop_norm'] = df_flow['Nguoi_nop'].apply(normalize_name)
    df_flow['Tai_khoan_norm'] = df_flow['Tai_khoan'].apply(normalize_account)
    df_flow['Ten_nha_dau_tu_norm'] = df_flow['Ten_nha_dau_tu'].apply(normalize_name)
    df_flow['So_tien_clean'] = df_flow['So_tien'].apply(parse_amount)
    
    # Lọc theo tùy chọn
    if exclude_internal:
        df_flow = df_flow[df_flow['Tu_chuyen_khoan'] != True]
    
    if transaction_type == 'Nộp':
        df_flow = df_flow[df_flow['NoP_Rut'] == 'Nộp']
    elif transaction_type == 'Rút':
        df_flow = df_flow[df_flow['NoP_Rut'] == 'Rút']
    
    # Đọc file nhóm danh tính (nếu có)
    df_group = None
    if group_file is not None:
        if group_file.name.endswith('.csv'):
            df_group = pd.read_csv(group_file)
        else:
            df_group = pd.read_excel(group_file)
        
        df_group['Ma_TK_norm'] = df_group['Ma_TK'].apply(normalize_account)
        df_group['Ten_norm'] = df_group['Ten'].apply(normalize_name)
    
    # Ghép dữ liệu
    if df_group is not None:
        df_merged = df_flow.merge(
            df_group[['Ma_TK_norm', 'STT_nhom', 'Ten', 'Moi_quan_he_voi_cong_ty']],
            left_on='Tai_khoan_norm',
            right_on='Ma_TK_norm',
            how='left'
        )
    else:
        df_merged = df_flow.copy()
        df_merged['STT_nhom'] = None
        df_merged['Ten'] = df_merged['Ten_nha_dau_tu']
        df_merged['Moi_quan_he_voi_cong_ty'] = 'Unknown'
    
    return df_merged

def create_edges(df, min_amount=0):
    """Tạo danh sách cạnh từ dữ liệu"""
    
    # Lọc theo ngưỡng tiền
    df_filtered = df[df['So_tien_clean'] >= min_amount]
    
    # Gộp theo nguồn - đích
    edges = df_filtered.groupby([
        'Nguoi_nop_norm', 
        'Tai_khoan_norm',
        'NoP_Rut',
        'STT_nhom'
    ]).agg({
        'So_tien_clean': 'sum',
        'So_lenh': 'sum',
        'Nguoi_nop': 'first',
        'Ten_nha_dau_tu': 'first',
        'Moi_quan_he_voi_cong_ty': 'first'
    }).reset_index()
    
    edges.columns = ['Nguon_norm', 'Dich_norm', 'Loai', 'STT_nhom', 
                     'Tong_tien', 'So_lenh', 'Nguon', 'Ten_dich', 'Moi_quan_he']
    
    return edges

# ==================== GRAPH ANALYSIS ====================

def build_graph(edges_df):
    """Xây dựng đồ thị từ edges"""
    G = nx.DiGraph()
    
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['Nguon_norm'],
            row['Dich_norm'],
            weight=row['Tong_tien'],
            orders=row['So_lenh'],
            group=row['STT_nhom'],
            relation=row['Moi_quan_he']
        )
        
        # Thêm thuộc tính cho node
        G.nodes[row['Nguon_norm']]['type'] = 'NGUON'
        G.nodes[row['Nguon_norm']]['label'] = row['Nguon']
        G.nodes[row['Nguon_norm']]['group'] = None
        
        G.nodes[row['Dich_norm']]['type'] = 'TAIKHOAN'
        G.nodes[row['Dich_norm']]['label'] = row['Ten_dich']
        G.nodes[row['Dich_norm']]['group'] = row['STT_nhom']
        G.nodes[row['Dich_norm']]['relation'] = row['Moi_quan_he']
    
    return G

def calculate_metrics(G):
    """Tính các chỉ số graph"""
    metrics = {}
    
    # Degree centrality
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Community detection
    G_undirected = G.to_undirected()
    communities = community_louvain.best_partition(G_undirected)
    
    for node in G.nodes():
        metrics[node] = {
            'in_degree': in_degree.get(node, 0),
            'out_degree': out_degree.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'community': communities.get(node, 0)
        }
    
    return metrics

def calculate_risk_score(row, edges_df):
    """Tính risk score cho mỗi cảnh báo"""
    score = 0
    
    # +3: Nguồn cấp cho ≥3 tài khoản khác nhóm
    if row.get('Loai_canh_bao') == 'Cross-group':
        unique_groups = edges_df[edges_df['Nguon_norm'] == row.get('Nguon_norm', '')]['STT_nhom'].nunique()
        if unique_groups >= 3:
            score += 3
    
    # +2: Quan hệ nội bộ/CĐL/NBTT
    sensitive_relations = ['NBTT', 'CĐL', 'nội bộ', 'người nội bộ', 'cổ đông lớn']
    if any(rel in str(row.get('Moi_quan_he', '')).lower() for rel in [r.lower() for r in sensitive_relations]):
        score += 2
    
    # +1: Tổng tiền ≥ 5 tỷ
    if row.get('Tong_tien', 0) >= 5_000_000_000:
        score += 1
    
    return min(score, 10)  # Cap at 10

# ==================== PHÁT HIỆN CẢNH BÁO ====================

def detect_alerts(edges_df, G, metrics):
    """Phát hiện các trường hợp cảnh báo"""
    alerts = []
    
    # 1. Cùng nguồn trong cùng nhóm
    same_group = edges_df[edges_df['STT_nhom'].notna()].groupby(['Nguon_norm', 'STT_nhom']).agg({
        'Dich_norm': lambda x: list(x),
        'Tong_tien': 'sum',
        'So_lenh': 'sum',
        'Nguon': 'first',
        'Moi_quan_he': 'first'
    }).reset_index()
    
    same_group = same_group[same_group['Dich_norm'].apply(len) >= 2]
    
    for _, row in same_group.iterrows():
        alert_data = {
            'Loai_canh_bao': 'Same-group',
            'Nguon': row['Nguon'],
            'Nguon_norm': row['Nguon_norm'],
            'STT_nhom': row['STT_nhom'],
            'So_tai_khoan': len(row['Dich_norm']),
            'Tong_tien': row['Tong_tien'],
            'So_lenh': row['So_lenh'],
            'Moi_quan_he': row['Moi_quan_he'],
            'Chi_tiet': f"Nguồn cấp tiền cho {len(row['Dich_norm'])} TK cùng nhóm {row['STT_nhom']}"
        }
        alert_data['Risk_Score'] = calculate_risk_score(alert_data, edges_df)
        alerts.append(alert_data)
    
    # 2. Cross-group (cùng nguồn khác nhóm)
    cross_group = edges_df[edges_df['STT_nhom'].notna()].groupby('Nguon_norm').agg({
        'STT_nhom': lambda x: list(x.unique()),
        'Dich_norm': lambda x: list(x),
        'Tong_tien': 'sum',
        'So_lenh': 'sum',
        'Nguon': 'first',
        'Moi_quan_he': 'first'
    }).reset_index()
    
    cross_group = cross_group[cross_group['STT_nhom'].apply(lambda x: len(x) >= 2)]
    
    for _, row in cross_group.iterrows():
        alert_data = {
            'Loai_canh_bao': 'Cross-group',
            'Nguon': row['Nguon'],
            'Nguon_norm': row['Nguon_norm'],
            'STT_nhom': f"Xuyên {len(row['STT_nhom'])} nhóm",
            'So_tai_khoan': len(row['Dich_norm']),
            'Tong_tien': row['Tong_tien'],
            'So_lenh': row['So_lenh'],
            'Moi_quan_he': row['Moi_quan_he'],
            'Chi_tiet': f"⚠️ CẢNH BÁO: Nguồn cấp cho {len(row['Dich_norm'])} TK thuộc {len(row['STT_nhom'])} nhóm khác nhau: {row['STT_nhom']}"
        }
        alert_data['Risk_Score'] = calculate_risk_score(alert_data, edges_df)
        alerts.append(alert_data)
    
    # 3. Tài khoản trung gian (high betweenness)
    high_betweenness = [(node, metrics[node]['betweenness']) 
                        for node in G.nodes() 
                        if G.nodes[node]['type'] == 'TAIKHOAN' 
                        and metrics[node]['betweenness'] > 0.01]
    
    high_betweenness.sort(key=lambda x: x[1], reverse=True)
    
    for node, score in high_betweenness[:10]:
        alert_data = {
            'Loai_canh_bao': 'Trung gian',
            'Nguon': G.nodes[node]['label'],
            'Nguon_norm': node,
            'STT_nhom': G.nodes[node].get('group'),
            'So_tai_khoan': G.out_degree(node),
            'Tong_tien': metrics[node]['in_degree'],
            'So_lenh': 0,
            'Moi_quan_he': G.nodes[node].get('relation'),
            'Chi_tiet': f"TK trung gian có betweenness = {score:.4f}"
        }
        alert_data['Risk_Score'] = calculate_risk_score(alert_data, edges_df)
        alerts.append(alert_data)
    
    return pd.DataFrame(alerts)

# ==================== VISUALIZATION ====================

def create_pyvis_graph(G, metrics, output_file='graph.html'):
    """Tạo graph tương tác với PyVis"""
    net = Network(height='750px', width='100%', directed=True, notebook=False)
    
    # Màu cho nhóm
    group_colors = {
        None: '#CCCCCC',
        1: '#FF6B6B',
        2: '#4ECDC4',
        3: '#45B7D1',
        4: '#FFA07A',
        5: '#98D8C8',
        6: '#F7DC6F',
        7: '#BB8FCE',
        8: '#85C1E2',
    }
    
    # Thêm nodes
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        label = G.nodes[node]['label']
        group = G.nodes[node].get('group')
        
        size = 10 + metrics[node]['in_degree'] / 1_000_000_000 * 5
        
        color = group_colors.get(group, '#CCCCCC')
        shape = 'dot' if node_type == 'TAIKHOAN' else 'square'
        
        title = f"""
        <b>{label}</b><br>
        Loại: {node_type}<br>
        Nhóm: {group if group else 'N/A'}<br>
        Tiền vào: {metrics[node]['in_degree']:,.0f} VNĐ<br>
        Tiền ra: {metrics[node]['out_degree']:,.0f} VNĐ<br>
        Betweenness: {metrics[node]['betweenness']:.4f}<br>
        Community: {metrics[node]['community']}
        """
        
        net.add_node(node, label=label[:20], title=title, size=size, 
                     color=color, shape=shape)
    
    # Thêm edges
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        width = 1 + weight / 10_000_000_000 * 5
        
        title = f"Tổng tiền: {weight:,.0f} VNĐ<br>Số lệnh: {edge[2]['orders']}"
        
        net.add_edge(edge[0], edge[1], value=width, title=title)
    
    # Cấu hình physics
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    net.save_graph(output_file)
    return output_file

def create_plotly_graph(G, metrics):
    """Tạo graph với Plotly"""
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )
        )
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[],
            color=[],
            colorbar=dict(
                thickness=15,
                title='Tổng tiền (tỷ)',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([G.nodes[node]['label'][:15]])
        node_trace['marker']['size'] += tuple([10 + metrics[node]['in_degree'] / 1_000_000_000 * 5])
        node_trace['marker']['color'] += tuple([metrics[node]['in_degree'] / 1_000_000_000])
    
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='Mạng quan hệ luồng tiền',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700
                    ))
    
    return fig

# ==================== EXPORT FUNCTIONS ====================

def export_to_excel(edges_df, alerts_df, metrics_dict, G):
    """Xuất dữ liệu ra Excel nhiều sheet"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Edges
        edges_df.to_excel(writer, sheet_name='Edges', index=False)
        
        # Sheet 2: Alerts với Risk Score
        if not alerts_df.empty:
            alerts_export = alerts_df.sort_values('Risk_Score', ascending=False)
            alerts_export.to_excel(writer, sheet_name='Alerts', index=False)
        
        # Sheet 3: Node metrics
        nodes_data = []
        for node in G.nodes():
            nodes_data.append({
                'Node_ID': node,
                'Label': G.nodes[node]['label'],
                'Type': G.nodes[node]['type'],
                'Group': G.nodes[node].get('group'),
                'In_Degree': metrics_dict[node]['in_degree'],
                'Out_Degree': metrics_dict[node]['out_degree'],
                'Betweenness': metrics_dict[node]['betweenness'],
                'Community': metrics_dict[node]['community']
            })
        
        pd.DataFrame(nodes_data).to_excel(writer, sheet_name='Nodes', index=False)
        
        # Sheet 4: Summary statistics
        summary = {
            'Metric': ['Total Nodes', 'Total Edges', 'Total Amount (VNĐ)', 
                      'Number of Communities', 'Number of Alerts'],
            'Value': [
                G.number_of_nodes(),
                G.number_of_edges(),
                edges_df['Tong_tien'].sum(),
                len(set(metrics_dict[n]['community'] for n in G.nodes())),
                len(alerts_df)
            ]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output

# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(page_title="Money Flow Analyzer", layout="wide", page_icon="💰")
    
    st.title("🔍 Công cụ Phân tích Luồng Tiền & Quan hệ Tài khoản")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        st.subheader("📂 Upload dữ liệu")
        flow_file = st.file_uploader("File luồng tiền (CSV/Excel)", 
                                     type=['csv', 'xlsx'],
                                     help="File chứa: Nguoi_nop, Tai_khoan, So_tien, NoP_Rut...")
        
        group_file = st.file_uploader("File nhóm danh tính (tùy chọn)", 
                                      type=['csv', 'xlsx'],
                                      help="File chứa: STT_nhom, Ma_TK, Ten, Moi_quan_he...")
        
        st.markdown("---")
        st.subheader("🔧 Bộ lọc")
        
        exclude_internal = st.checkbox("Loại trừ giao dịch nội bộ", value=True,
                                       help="Loại bỏ dòng Tu_chuyen_khoan = TRUE")
        
        transaction_type = st.selectbox("Loại giao dịch", 
                                        ['Cả hai', 'Nộp', 'Rút'])
        
        min_amount = st.number_input("Ngưỡng tiền tối thiểu (VNĐ)", 
                                     min_value=0, 
                                     value=0, 
                                     step=100000000,
                                     format="%d")
        
        st.markdown("---")
        st.subheader("📊 Tùy chọn hiển thị")
        viz_type = st.radio("Loại biểu đồ", ['PyVis (Interactive)', 'Plotly'])
    
    # Main content
    if flow_file is not None:
        try:
            # Load data
            with st.spinner("Đang xử lý dữ liệu..."):
                df = load_and_process_data(flow_file, group_file, exclude_internal, transaction_type)
                edges_df = create_edges(df, min_amount)
                
                if len(edges_df) == 0:
                    st.warning("⚠️ Không có dữ liệu sau khi lọc. Vui lòng điều chỉnh bộ lọc.")
                    return
                
                G = build_graph(edges_df)
                metrics = calculate_metrics(G)
                alerts_df = detect_alerts(edges_df, G, metrics)
            
            st.success(f"✅ Đã tải {len(df)} giao dịch, tạo {G.number_of_nodes()} nodes và {G.number_of_edges()} edges")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Biểu đồ", "⚠️ Cảnh báo", "📊 Thống kê", "💾 Xuất dữ liệu"])
            
            with tab1:
                st.subheader("Mạng quan hệ luồng tiền")
                
                if viz_type == 'PyVis (Interactive)':
                    html_file = create_pyvis_graph(G, metrics, 'temp_graph.html')
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800, scrolling=True)
                    
                    # Download HTML
                    st.download_button(
                        label="📥 Tải biểu đồ HTML",
                        data=html_content,
                        file_name=f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                else:
                    fig = create_plotly_graph(G, metrics)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download PNG
                    img_bytes = fig.to_image(format="png", width=1920, height=1080)
                    st.download_button(
                        label="📥 Tải biểu đồ PNG",
                        data=img_bytes,
                        file_name=f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
            
            with tab2:
                st.subheader("🚨 Danh sách Cảnh báo")
                
                if not alerts_df.empty:
                    # Thêm màu cho risk score
                    def highlight_risk(val):
                        if val >= 7:
                            return 'background-color: #ff4444; color: white; font-weight: bold'
                        elif val >= 4:
                            return 'background-color: #ffaa00; font-weight: bold'
                        else:
                            return 'background-color: #ffff99'
                    
                    # Sắp xếp theo Risk Score
                    alerts_display = alerts_df.sort_values('Risk_Score', ascending=False).copy()
                    
                    # Format số tiền
                    alerts_display['Tong_tien_display'] = alerts_display['Tong_tien'].apply(
                        lambda x: f"{x:,.0f} VNĐ"
                    )
                    
                    # Hiển thị bảng với style
                    styled_df = alerts_display[[
                        'Risk_Score', 'Loai_canh_bao', 'Nguon', 'STT_nhom', 
                        'So_tai_khoan', 'Tong_tien_display', 'Moi_quan_he', 'Chi_tiet'
                    ]].style.applymap(highlight_risk, subset=['Risk_Score'])
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Thống kê nhanh
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tổng cảnh báo", len(alerts_df))
                    with col2:
                        high_risk = len(alerts_df[alerts_df['Risk_Score'] >= 7])
                        st.metric("Rủi ro cao (≥7)", high_risk, delta_color="inverse")
                    with col3:
                        cross_group = len(alerts_df[alerts_df['Loai_canh_bao'] == 'Cross-group'])
                        st.metric("Cross-group", cross_group)
                    with col4:
                        total_alert_money = alerts_df['Tong_tien'].sum()
                        st.metric("Tổng tiền cảnh báo", f"{total_alert_money/1e9:.1f}B VNĐ")
                else:
                    st.info("Không phát hiện cảnh báo nào")
            
            with tab3:
                st.subheader("📊 Thống kê tổng quan")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 10 Nguồn tiền lớn nhất**")
                    top_sources = edges_df.groupby('Nguon')['Tong_tien'].sum().sort_values(ascending=False).head(10)
                    st.dataframe(top_sources.apply(lambda x: f"{x:,.0f} VNĐ"))
                    
                    st.markdown("**Top 10 Tài khoản nhận nhiều nhất**")
                    top_accounts = edges_df.groupby('Ten_dich')['Tong_tien'].sum().sort_values(ascending=False).head(10)
                    st.dataframe(top_accounts.apply(lambda x: f"{x:,.0f} VNĐ"))
                
                with col2:
                    st.markdown("**Top 10 Tài khoản trung gian (Betweenness)**")
                    betweenness_list = [(G.nodes[n]['label'], metrics[n]['betweenness']) 
                                       for n in G.nodes() if G.nodes[n]['type'] == 'TAIKHOAN']
                    betweenness_list.sort(key=lambda x: x[1], reverse=True)
                    st.dataframe(pd.DataFrame(betweenness_list[:10], 
                                            columns=['Tài khoản', 'Betweenness Score']))
                    
                    st.markdown("**Phân bố Community**")
                    community_dist = pd.Series([metrics[n]['community'] for n in G.nodes()]).value_counts()
                    st.bar_chart(community_dist)
                
                # Network statistics
                st.markdown("---")
                st.markdown("**Chỉ số mạng**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Nodes", G.number_of_nodes())
                with col2:
                    st.metric("Edges", G.number_of_edges())
                with col3:
                    st.metric("Tổng tiền", f"{edges_df['Tong_tien'].sum()/1e9:.1f}B")
                with col4:
                    components = nx.number_weakly_connected_components(G)
                    st.metric("Connected Components", components)
                with col5:
                    communities_count = len(set(metrics[n]['community'] for n in G.nodes()))
                    st.metric("Communities", communities_count)
            
            with tab4:
                st.subheader("💾 Xuất dữ liệu")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Xuất Excel (Tất cả sheets)**")
                    excel_data = export_to_excel(edges_df, alerts_df, metrics, G)
                    
                    st.download_button(
                        label="📥 Tải file Excel đầy đủ",
                        data=excel_data,
                        file_name=f"money_flow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.info("File Excel bao gồm 4 sheets:\n- **Edges**: Danh sách cạnh\n- **Alerts**: Cảnh báo có Risk Score\n- **Nodes**: Thông tin nodes\n- **Summary**: Tóm tắt")
                
                with col2:
                    st.markdown("**📄 Xuất CSV riêng lẻ**")
                    
                    # Edges CSV
                    edges_csv = edges_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Tải Edges.csv",
                        data=edges_csv,
                        file_name=f"edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Alerts CSV
                    if not alerts_df.empty:
                        alerts_csv = alerts_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải Alerts.csv",
                            data=alerts_csv,
                            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                st.markdown("---")
                
                # Preview data
                with st.expander("👁️ Xem trước dữ liệu Edges"):
                    st.dataframe(edges_df.head(20), use_container_width=True)
                
                if not alerts_df.empty:
                    with st.expander("👁️ Xem trước dữ liệu Alerts"):
                        st.dataframe(alerts_df.head(20), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Lỗi xử lý dữ liệu: {str(e)}")
            st.exception(e)
    
    else:
        # Demo mode hoặc hướng dẫn
        st.info("👆 Vui lòng upload file dữ liệu ở sidebar để bắt đầu phân tích")
        
        with st.expander("📖 Hướng dẫn sử dụng"):
            st.markdown("""
            ### Cách sử dụng tool:
            
            1. **Upload file luồng tiền** (bắt buộc):
               - Cột cần có: `Nguoi_nop`, `Tai_khoan`, `So_tien`, `NoP_Rut`, `Tu_chuyen_khoan`, `So_lenh`, `Ten_nha_dau_tu`, `CTCK`
            
            2. **Upload file nhóm danh tính** (tùy chọn):
               - Cột cần có: `STT_nhom`, `Ma_TK`, `Ten`, `Moi_quan_he_voi_cong_ty`
            
            3. **Cấu hình bộ lọc**:
               - Loại trừ giao dịch nội bộ
               - Chọn loại giao dịch (Nộp/Rút/Cả hai)
               - Đặt ngưỡng tiền tối thiểu
            
            4. **Phân tích kết quả**:
               - **Tab Biểu đồ**: Xem mạng quan hệ tương tác, tải HTML/PNG
               - **Tab Cảnh báo**: Danh sách cảnh báo có Risk Score (0-10)
               - **Tab Thống kê**: Top nguồn tiền, tài khoản trung gian, community
               - **Tab Xuất dữ liệu**: Tải Excel đầy đủ hoặc CSV riêng lẻ
            
            ### Ý nghĩa Risk Score:
            - **0-3**: Rủi ro thấp (màu vàng nhạt)
            - **4-6**: Rủi ro trung bình (màu cam)
            - **7-10**: Rủi ro cao (màu đỏ, cần kiểm tra kỹ)
            
            ### Các loại cảnh báo:
            - **Same-group**: Cùng nguồn tiền trong cùng nhóm danh tính (liên hệ rất mạnh)
            - **Cross-group**: Cùng nguồn tiền xuyên nhiều nhóm (cảnh báo nghiêm trọng)
            - **Trung gian**: Tài khoản có vai trò trung gian chuyển tiền
            """)
        
        with st.expander("🧪 Chạy với dữ liệu mẫu"):
            st.markdown("""
            Dữ liệu mẫu được tích hợp sẵn để demo. Tuy nhiên, để sử dụng đầy đủ tính năng,
            vui lòng upload file dữ liệu thực tế của bạn.
            """)

if __name__ == "__main__":
    main()
