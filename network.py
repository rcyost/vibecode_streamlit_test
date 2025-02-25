#%%
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import tempfile
import json
from streamlit_elements import elements, dashboard, mui, html
import io

# Function to create network visualization using Plotly
def create_network(adj_matrix, node_labels=None):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph)
    
    # Create layout for nodes
    pos = nx.spring_layout(G)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_texts = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = adj_matrix[edge[0]][edge[1]]
        edge_weights.append(weight)
        source_label = node_labels[edge[0]] if node_labels else f"Node {edge[0]}"
        target_label = node_labels[edge[1]] if node_labels else f"Node {edge[1]}"
        edge_texts.append(f"{source_label} → {target_label}: {weight:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='black'),
        hoverinfo='text',
        text=edge_texts,
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_texts = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = node_labels[node] if node_labels else f"Node {node}"
        node_texts.append(label)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_texts,
        textposition="top center",
        textfont=dict(
            color='black',  # Set node label color to black
            size=12        # Optional: adjust text size if needed
        ),
        marker=dict(
            color='red',
            size=20,
            line=dict(width=2)
        )
    )
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    # Update layout for better interactivity
    fig.update_layout(
        dragmode='pan',
        width=800,
        height=600
    )
    
    return fig

# Add this function near the top of the file after imports
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# Add after other function definitions
def get_current_matrix():
    """Get the current adjacency matrix if it exists in session state"""
    if 'current_matrix' in st.session_state and 'current_labels' in st.session_state:
        return pd.DataFrame(
            st.session_state.current_matrix,
            index=st.session_state.current_labels,
            columns=st.session_state.current_labels
        )
    return None

# Add this CSS to ensure centered network visualization
# Add near the top after the imports
def add_network_css():
    st.markdown("""
        <style>
        .network-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
        }
        iframe {
            border: none !important;
            margin: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Add this function after the other utility functions
def get_example_network():
    """Create an example network with Trump administration officials"""
    # Trump administration key figures
    labels = ["Donald Trump", "Mike Pence", "Steve Bannon", "Jared Kushner", "Ivanka Trump"]
    
    # Create example adjacency matrix with meaningful connections
    example_matrix = pd.DataFrame([
        [0.0, 0.8, 0.7, 0.9, 0.9],  # Trump's influence
        [0.6, 0.0, 0.5, 0.6, 0.6],  # Pence's influence
        [0.7, 0.4, 0.0, 0.5, 0.5],  # Bannon's influence
        [0.8, 0.5, 0.4, 0.0, 0.9],  # Kushner's influence
        [0.8, 0.5, 0.4, 0.9, 0.0],  # Ivanka's influence
    ], index=labels, columns=labels)
    
    return example_matrix

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Network Analysis App",
    layout="wide"
)

# Add a title to the app
st.title("Network Analysis Dashboard")

# Update sidebar section
with st.sidebar:
    st.header("Input Data")
    
    # Initialize session state for example data if not exists
    if 'example_data' not in st.session_state:
        # Automatically load example data on first run
        st.session_state.example_data = get_example_network()
        
    upload_method = st.radio(
        "Choose input method",
        ["Load Example", "Upload CSV", "Manual Input"],  # Reordered to make example first
        index=0  # Default to Load Example
    )
    
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload adjacency matrix (CSV)",
            type=["csv"]
        )
        # Clear example data when switching to upload
        st.session_state.example_data = None
        
    elif upload_method == "Load Example":
        st.write("Example: Political Influence Network")
        st.write("This example shows influence relationships between different political actors.")
        if st.session_state.example_data is None:
            if st.button("Load Example Network"):
                st.session_state.example_data = get_example_network()
        # Convert example data to CSV format for the uploader
        if st.session_state.example_data is not None:
            csv_data = convert_df_to_csv(st.session_state.example_data)
            uploaded_file = io.BytesIO(csv_data)
    else:
        # Clear example data when switching to manual
        st.session_state.example_data = None
        size = st.number_input(
            "Matrix size",
            min_value=2,
            max_value=20,
            value=3
        )
    
    # Add download section to sidebar
    st.markdown("---")  # Add a visual separator
    st.header("Download Data")
    current_matrix = get_current_matrix()
    if current_matrix is not None:
        csv = convert_df_to_csv(current_matrix)
        st.download_button(
            label="Download Current Matrix",
            data=csv,
            file_name="adjacency_matrix.csv",
            mime="text/csv",
        )
    else:
        st.write("Generate a network to enable download")

# Move matrix input to main area
if upload_method == "Manual Input":
    # Create tabs for instructions, node labels, matrix input, and network
    tab0, tab1, tab2, tab3 = st.tabs(["📖 Instructions", "📝 Node Labels", "🔢 Adjacency Matrix", "🕸️ Network Analysis"])
    
    with tab0:
        st.header("How to Use This Network Analysis Tool")
        
        st.subheader("Step 1: Enter Node Labels")
        st.write("""
        - Go to the 'Node Labels' tab
        - Enter meaningful names for each node in your network
        - Default labels are provided but can be changed to any text
        - These labels will help identify connections in your network
        """)
        
        st.subheader("Step 2: Create Adjacency Matrix")
        st.write("""
        - Navigate to the 'Adjacency Matrix' tab
        - Enter values between 0 and 1 to represent connection strengths:
            - 0 = No connection
            - 1 = Strongest connection
            - Any decimal between 0 and 1 represents partial connection strength
        - Each row represents connections FROM that node
        - Each column represents connections TO that node
        - Click 'Generate Network' when your matrix is complete
        """)
        
        st.subheader("Step 3: View Network Analysis")
        st.write("""
        - After generating, you'll be taken to the 'Network Analysis' tab
        - The visualization shows:
            - Nodes with your custom labels
            - Connections with varying thicknesses based on strength
            - Hover over nodes and edges to see details
        - Network metrics are displayed below the visualization
        """)
        
        st.subheader("Additional Features")
        st.write("""
        - Use the download button in the sidebar to save your adjacency matrix
        - You can upload a previously saved matrix using the CSV upload option
        - The network visualization is interactive:
            - Drag nodes to rearrange
            - Zoom in/out with mouse wheel
            - Pan by clicking and dragging the background
        """)
        
        st.info("💡 Tip: Make sure to give your nodes meaningful labels - this will make the network visualization much more useful!")

    with tab1:
        st.write("Enter node labels:")
        # Organize node labels in rows of 5
        node_labels = []
        for i in range(0, size, 5):
            cols = st.columns(min(5, size - i))
            for j, col in enumerate(cols):
                with col:
                    idx = i + j
                    label = st.text_input(
                        f"Node {idx} name",
                        value=f"Node {idx}",
                        key=f"node_label_{idx}"
                    )
                    node_labels.append(label)
    
    with tab2:
        st.write("Enter adjacency matrix values (between 0 and 1):")
        
        # Move Generate Network button to top
        generate_network = st.button("Generate Network", type="primary")
        
        # Create expandable sections for each row of the matrix
        matrix_input = []
        for i in range(size):
            with st.expander(f"Row {i + 1}: {node_labels[i]}", expanded=True):
                row = []
                # Organize matrix inputs in rows of 5
                for j in range(0, size, 5):
                    cols = st.columns(min(5, size - j))
                    for k, col in enumerate(cols):
                        idx = j + k
                        with col:
                            val = st.text_input(
                                f"{node_labels[idx]}",
                                value="0",
                                key=f"matrix_{i}_{idx}",
                                help=f"Connection: {node_labels[i]} → {node_labels[idx]}"
                            )
                            try:
                                val_float = float(val)
                                if val_float < 0 or val_float > 1:
                                    st.error("Value must be between 0 and 1")
                                    val_float = 0
                            except ValueError:
                                st.error("Invalid number")
                                val_float = 0
                            row.append(val_float)
                matrix_input.append(row)
        
        # Move button logic here
        if generate_network:
            adj_matrix = pd.DataFrame(matrix_input).values
            # Store current matrix in session state
            st.session_state.current_matrix = adj_matrix
            st.session_state.current_labels = node_labels
            
            # Show matrix in an expander
            with st.expander("View Adjacency Matrix", expanded=False):
                df_display = pd.DataFrame(
                    adj_matrix,
                    index=node_labels,
                    columns=node_labels
                )
                st.dataframe(df_display.style.format("{:.3f}"))
            
            # Store in session state that network should be displayed
            st.session_state.show_network = True
            st.session_state.network_data = create_network(adj_matrix, node_labels)
            # Update to use new query_params
            st.query_params["tab"] = "network"
    
    with tab3:
        if 'show_network' in st.session_state and st.session_state.show_network:
            # Display network
            st.plotly_chart(st.session_state.network_data, use_container_width=True)
            
            # Add network metrics
            if 'current_matrix' in st.session_state:
                G = nx.from_numpy_array(st.session_state.current_matrix)
                
                st.header("Network Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Number of Nodes", len(G.nodes()))
                with col2:
                    st.metric("Number of Edges", len(G.edges()))
                with col3:
                    st.metric("Average Degree", float(sum(dict(G.degree()).values()))/len(G.nodes()))
        else:
            st.info("Generate a network in the Adjacency Matrix tab to view the analysis")

# Update CSV upload section
if (upload_method == "Upload CSV" and uploaded_file is not None) or st.session_state.example_data is not None:
    # Create tabs for instructions and other sections
    tab0, tab1, tab2, tab3 = st.tabs(["📖 Instructions", "📝 Node Labels", "🔢 Adjacency Matrix", "🕸️ Network Analysis"])
    
    try:
        # Handle both regular CSV and example data
        if st.session_state.example_data is not None:
            df = st.session_state.example_data
            adj_matrix = df.values
            node_labels = list(df.index)
            
            # Automatically show the network for example data
            if 'show_network' not in st.session_state:
                st.session_state.show_network = True
                st.session_state.network_data = create_network(adj_matrix, node_labels)
                st.session_state.current_matrix = adj_matrix
                st.session_state.current_labels = node_labels
        else:
            # Regular CSV handling
            df = pd.read_csv(uploaded_file, index_col=0)
            node_labels = list(df.index)
            adj_matrix = df.values
            adj_matrix = pd.DataFrame(adj_matrix).apply(pd.to_numeric, errors='coerce').fillna(0).values

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            st.error("The uploaded matrix must be square (same number of rows and columns)")
            st.stop()
            
        size = len(adj_matrix)
        
        with tab0:
            st.header("How to Use This Network Analysis Tool")
            
            st.subheader("Step 1: Load Example")
            st.write("""
            - The example network has been automatically loaded ✅
            - This political influence network demonstrates relationships between key actors
            """)
            
            st.subheader("Step 2: Enter Node Labels")
            st.write("""
            - Go to the 'Node Labels' tab
            - Enter meaningful names for each node in your network
            - Default labels are provided but can be changed to any text
            - These labels will help identify connections in your network
            """)
            
            st.subheader("Step 3: Generate Network")
            st.write("""
            - Navigate to the 'Adjacency Matrix' tab
            - Review your uploaded matrix
            - Click 'Generate Network' to create the visualization
            """)
            
            st.subheader("Step 4: View Network Analysis")
            st.write("""
            - After generating, you'll be taken to the 'Network Analysis' tab
            - The visualization shows:
                - Nodes with your custom labels
                - Connections with varying thicknesses based on strength
                - Hover over nodes and edges to see details
            - Network metrics are displayed below the visualization
            """)
            
            st.subheader("Additional Features")
            st.write("""
            - Use the download button in the sidebar to save your adjacency matrix
            - The network visualization is interactive:
                - Drag nodes to rearrange
                - Zoom in/out with mouse wheel
                - Pan by clicking and dragging the background
            """)
            
            st.info("💡 Tip: Make sure to give your nodes meaningful labels - this will make the network visualization much more useful!")

        with tab1:
            st.write("Node labels from CSV:")
            csv_node_labels = []
            # Organize node labels in rows of 5
            for i in range(0, size, 5):
                cols = st.columns(min(5, size - i))
                for j, col in enumerate(cols):
                    with col:
                        idx = i + j
                        # Pre-populate with labels from CSV
                        label = st.text_input(
                            f"Node {idx} name",
                            value=node_labels[idx] if idx < len(node_labels) else f"Node {idx}",
                            key=f"csv_node_label_{idx}"
                        )
                        csv_node_labels.append(label)
        
        with tab2:
            # Move Generate Network button to top
            generate_network = st.button("Generate Network", type="primary")
            
            st.write("Uploaded Adjacency Matrix:")
            df_display = pd.DataFrame(
                adj_matrix,
                index=csv_node_labels,
                columns=csv_node_labels
            )
            # Update the format string to handle any data type
            st.dataframe(df_display.style.format(lambda x: f"{float(x):.3f}" if isinstance(x, (int, float)) else str(x)))
            
            # Store current matrix in session state
            st.session_state.current_matrix = adj_matrix
            st.session_state.current_labels = csv_node_labels
            
            # Move button logic here
            if generate_network:
                # Store in session state that network should be displayed
                st.session_state.show_network = True
                st.session_state.network_data = create_network(adj_matrix, csv_node_labels)
                # Update to use new query_params
                st.query_params["tab"] = "network"
        
        with tab3:
            if 'show_network' in st.session_state and st.session_state.show_network:
                # Display network
                st.plotly_chart(st.session_state.network_data, use_container_width=True)
                
                # Add network metrics
                if 'current_matrix' in st.session_state:
                    G = nx.from_numpy_array(st.session_state.current_matrix)
                    
                    st.header("Network Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Nodes", len(G.nodes()))
                    with col2:
                        st.metric("Number of Edges", len(G.edges()))
                    with col3:
                        st.metric("Average Degree", float(sum(dict(G.degree()).values()))/len(G.nodes()))
            else:
                st.info("Generate a network in the Adjacency Matrix tab to view the analysis")

    except Exception as e:
        st.error(f"""
        Error reading CSV file. Please ensure:
        - The file has node labels in the first column and header row
        - The remaining cells contain only numbers
        - All values are between 0 and 1
        - The matrix is square (same number of rows and columns)
        
        Technical details: {str(e)}
        """)
        st.stop()
