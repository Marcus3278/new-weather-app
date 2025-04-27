"""
Data Sharing Components for Collaborative Data Exploration

This module provides components for real-time data sharing and collaborative
exploration in Streamlit applications.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid

def render_shared_data_explorer(collaboration, data=None):
    """
    Render a shared data explorer component.
    
    Args:
        collaboration: Streamlit collaboration client
        data: Optional pandas DataFrame to share
    """
    # Two-column layout for upload/share and received data
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Share Your Data")
        
        # Option to share current dataset if available
        if data is not None:
            st.write(f"Current dataset: {data.shape[0]} rows × {data.shape[1]} columns")
            st.dataframe(data.head(5))
            
            # Share data button
            if st.button("Share Current Dataset"):
                # Convert to JSON for sharing
                data_json = data.to_json(orient="split", date_format="iso")
                
                # Prepare metadata
                metadata = {
                    "rows": data.shape[0],
                    "columns": data.shape[1],
                    "column_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
                    "shared_at": datetime.now().isoformat()
                }
                
                # Share with collaborators
                collaboration.share_data("dataset", {
                    "name": "Shared Dataset",
                    "data": data_json,
                    "metadata": metadata
                })
                
                st.success("✅ Dataset shared with collaborators!")
        
        # Option to upload a new file to share
        st.subheader("Or Upload New Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                uploaded_data = pd.read_csv(uploaded_file)
                
                st.write(f"Uploaded dataset: {uploaded_data.shape[0]} rows × {uploaded_data.shape[1]} columns")
                st.dataframe(uploaded_data.head(5))
                
                # Share uploaded data button
                if st.button("Share Uploaded Dataset"):
                    # Convert to JSON for sharing
                    data_json = uploaded_data.to_json(orient="split", date_format="iso")
                    
                    # Prepare metadata
                    metadata = {
                        "filename": uploaded_file.name,
                        "rows": uploaded_data.shape[0],
                        "columns": uploaded_data.shape[1],
                        "column_types": {col: str(dtype) for col, dtype in uploaded_data.dtypes.items()},
                        "shared_at": datetime.now().isoformat()
                    }
                    
                    # Share with collaborators
                    collaboration.share_data("uploaded_dataset", {
                        "name": f"Uploaded: {uploaded_file.name}",
                        "data": data_json,
                        "metadata": metadata
                    })
                    
                    st.success(f"✅ Uploaded dataset '{uploaded_file.name}' shared with collaborators!")
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
    
    with col2:
        st.subheader("Datasets From Collaborators")
        
        # Show shared datasets
        if "shared_datasets" in st.session_state and st.session_state.shared_datasets:
            # Show most recent datasets first
            datasets = list(st.session_state.shared_datasets.items())
            
            for dataset_id, dataset_info in reversed(datasets):
                metadata = dataset_info.get('metadata', {})
                
                with st.expander(f"{metadata.get('name', 'Unnamed Dataset')} - by {metadata.get('shared_by', 'Unknown')}"):
                    st.write(f"Shared at: {metadata.get('timestamp', 'Unknown time')}")
                    st.write(f"Rows: {metadata.get('rows', 'Unknown')}")
                    st.write(f"Columns: {metadata.get('columns', 'Unknown')}")
                    
                    # Button to use this dataset
                    if st.button(f"Use This Dataset", key=f"use_dataset_{dataset_id}"):
                        try:
                            # Load the dataset from JSON
                            dataset_json = dataset_info.get('data', {}).get('data')
                            if dataset_json:
                                loaded_data = pd.read_json(dataset_json, orient="split")
                                st.session_state.data = loaded_data
                                st.success(f"✅ Loaded shared dataset with {loaded_data.shape[0]} rows.")
                                st.experimental_rerun()
                            else:
                                st.error("Error: Dataset data not found.")
                        except Exception as e:
                            st.error(f"Error loading dataset: {e}")
        else:
            st.info("No shared datasets received yet.")

def render_shared_visualizations_viewer(collaboration):
    """
    Render a viewer component for shared visualizations.
    
    Args:
        collaboration: Streamlit collaboration client
    """
    st.subheader("Shared Visualizations")
    
    if "shared_visualizations" in st.session_state and st.session_state.shared_visualizations:
        # Show most recent visualizations first
        visualizations = list(st.session_state.shared_visualizations.items())
        
        for viz_id, viz_data in reversed(visualizations):
            metadata = viz_data.get('metadata', {})
            viz_type = metadata.get('type', 'Unknown')
            
            with st.expander(f"{metadata.get('title', 'Unnamed Visualization')} - by {metadata.get('shared_by', 'Unknown')}"):
                st.write(f"Shared at: {metadata.get('timestamp', 'Unknown time')}")
                st.write(f"Type: {viz_type}")
                
                # Try to render visualization based on type
                chart_data = viz_data.get('data', {})
                
                if viz_type == 'scatter':
                    # Recreate scatter plot
                    try:
                        x_col = chart_data.get('x')
                        y_col = chart_data.get('y')
                        data = pd.DataFrame(chart_data.get('data', []))
                        
                        if not data.empty and x_col in data.columns and y_col in data.columns:
                            fig = px.scatter(data, x=x_col, y=y_col, 
                                            title=metadata.get('title', f"Scatter Plot: {x_col} vs {y_col}"))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot render visualization: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering scatter plot: {e}")
                
                elif viz_type == 'histogram':
                    # Recreate histogram
                    try:
                        column = chart_data.get('column')
                        data = pd.Series(chart_data.get('data', {}))
                        
                        if not data.empty:
                            fig = px.histogram(data, title=metadata.get('title', f"Distribution of {column}"))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot render visualization: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering histogram: {e}")
                
                elif viz_type == 'boxplot':
                    # Recreate box plot
                    try:
                        column = chart_data.get('column')
                        data = pd.Series(chart_data.get('data', {}))
                        
                        if not data.empty:
                            fig = px.box(y=data, title=metadata.get('title', f"Box Plot of {column}"))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot render visualization: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering box plot: {e}")
                
                elif viz_type == 'heatmap':
                    # Recreate heatmap
                    try:
                        columns = chart_data.get('columns', [])
                        matrix_data = chart_data.get('matrix', {})
                        
                        if matrix_data and columns:
                            matrix_df = pd.DataFrame(matrix_data)
                            fig = px.imshow(
                                matrix_df,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                title=metadata.get('title', "Correlation Matrix")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot render visualization: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering heatmap: {e}")
                
                else:
                    st.info(f"This visualization type ({viz_type}) cannot be rendered directly.")
    else:
        st.info("No visualizations have been shared yet.")

def render_shared_analysis_viewer(collaboration):
    """
    Render a viewer component for shared analysis results.
    
    Args:
        collaboration: Streamlit collaboration client
    """
    st.subheader("Shared Analysis Results")
    
    if "shared_analyses" in st.session_state and st.session_state.shared_analyses:
        # Show most recent analyses first
        analyses = list(st.session_state.shared_analyses.items())
        
        for analysis_id, analysis_data in reversed(analyses):
            metadata = analysis_data.get('metadata', {})
            analysis_type = metadata.get('type', 'Unknown')
            
            with st.expander(f"{metadata.get('title', 'Unnamed Analysis')} - by {metadata.get('shared_by', 'Unknown')}"):
                st.write(f"Shared at: {metadata.get('timestamp', 'Unknown time')}")
                st.write(f"Type: {analysis_type}")
                
                # Try to render analysis based on type
                analysis_content = analysis_data.get('data', {})
                
                if analysis_type == 'basic_statistics':
                    # Render basic statistics
                    try:
                        stats_data = analysis_content
                        if stats_data:
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df)
                        else:
                            st.warning("Cannot render analysis: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering basic statistics: {e}")
                
                elif analysis_type == 'correlation_analysis':
                    # Render correlation analysis
                    try:
                        corr_data = analysis_content
                        columns = metadata.get('columns', [])
                        
                        if corr_data and columns:
                            corr_df = pd.DataFrame(corr_data)
                            
                            # Display the correlation matrix
                            fig = px.imshow(
                                corr_df,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                title="Correlation Matrix"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot render analysis: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering correlation analysis: {e}")
                
                elif analysis_type == 'cluster_analysis':
                    # Render cluster analysis
                    try:
                        cluster_data = analysis_content
                        
                        if cluster_data:
                            # Show cluster distribution
                            distribution = cluster_data.get('distribution', [])
                            if distribution:
                                st.write("Cluster Distribution:")
                                dist_df = pd.DataFrame(distribution)
                                st.dataframe(dist_df)
                            
                            # If 2D data is available, show cluster plot
                            centers = cluster_data.get('centers', [])
                            if centers and len(centers[0]) == 2:
                                centers_df = pd.DataFrame(centers, columns=['x', 'y'])
                                centers_df['Cluster'] = [f"Cluster {i}" for i in range(len(centers))]
                                
                                fig = px.scatter(centers_df, x='x', y='y', color='Cluster',
                                                title="Cluster Centers")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot render analysis: data format issue")
                    except Exception as e:
                        st.error(f"Error rendering cluster analysis: {e}")
                
                else:
                    st.info(f"This analysis type ({analysis_type}) cannot be rendered directly.")
    else:
        st.info("No analyses have been shared yet.")

def data_update_callback(data, session_state):
    """
    Handle incoming data updates from collaborators.
    
    Args:
        data: The data update message
        session_state: Streamlit session state
    """
    try:
        # Initialize shared datasets dict if not exists
        if "shared_datasets" not in session_state:
            session_state.shared_datasets = {}
        
        # Generate unique ID for this dataset
        dataset_id = str(uuid.uuid4())
        
        # Extract data content
        content = data.get('content', {})
        
        # Add metadata
        metadata = {
            "name": content.get('name', 'Unnamed Dataset'),
            "shared_by": data.get('user_name', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "type": data.get('data_type', 'Unknown'),
            "rows": content.get('metadata', {}).get('rows', 'Unknown'),
            "columns": content.get('metadata', {}).get('columns', 'Unknown')
        }
        
        # Store in session state
        session_state.shared_datasets[dataset_id] = {
            "data": content,
            "metadata": metadata
        }
        
        # Add notification to chat if available
        if 'collab_messages' in session_state:
            session_state.collab_messages.append({
                'type': 'info',
                'content': f"{metadata['shared_by']} shared a dataset with {metadata['rows']} rows and {metadata['columns']} columns."
            })
    except Exception as e:
        print(f"Error in data_update_callback: {e}")

def visualization_update_callback(data, session_state):
    """
    Handle incoming visualization updates from collaborators.
    
    Args:
        data: The visualization update message
        session_state: Streamlit session state
    """
    try:
        # Initialize shared visualizations dict if not exists
        if "shared_visualizations" not in session_state:
            session_state.shared_visualizations = {}
        
        # Generate unique ID for this visualization
        viz_id = str(uuid.uuid4())
        
        # Extract content
        content = data.get('content', {})
        chart_type = data.get('chart_type', 'unknown')
        chart_data = content.get('chart_data', {})
        
        # Add metadata
        metadata = {
            "title": chart_data.get('title', f"Visualization from {data.get('user_name', 'Unknown')}"),
            "shared_by": data.get('user_name', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "type": chart_type
        }
        
        # Store in session state
        session_state.shared_visualizations[viz_id] = {
            "data": chart_data,
            "metadata": metadata
        }
        
        # Add notification to chat if available
        if 'collab_messages' in session_state:
            session_state.collab_messages.append({
                'type': 'info',
                'content': f"{metadata['shared_by']} shared a {chart_type} visualization: {metadata['title']}"
            })
    except Exception as e:
        print(f"Error in visualization_update_callback: {e}")

def analysis_update_callback(data, session_state):
    """
    Handle incoming analysis updates from collaborators.
    
    Args:
        data: The analysis update message
        session_state: Streamlit session state
    """
    try:
        # Initialize shared analyses dict if not exists
        if "shared_analyses" not in session_state:
            session_state.shared_analyses = {}
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Extract content
        content = data.get('content', {})
        analysis_type = data.get('analysis_type', 'unknown')
        analysis_data = content.get('results', {})
        
        # Extract parameters
        parameters = content.get('parameters', {})
        
        # Add metadata
        metadata = {
            "title": analysis_data.get('title', f"Analysis from {data.get('user_name', 'Unknown')}"),
            "shared_by": data.get('user_name', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "type": analysis_type,
            "dataset_name": parameters.get('dataset', 'Unknown dataset')
        }
        
        # Store in session state
        session_state.shared_analyses[analysis_id] = {
            "data": analysis_data,
            "metadata": metadata,
            "parameters": parameters
        }
        
        # Add notification to chat if available
        if 'collab_messages' in session_state:
            session_state.collab_messages.append({
                'type': 'info',
                'content': f"{metadata['shared_by']} shared a {analysis_type} analysis: {metadata['title']}"
            })
    except Exception as e:
        print(f"Error in analysis_update_callback: {e}")