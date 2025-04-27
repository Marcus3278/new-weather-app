"""
Suggestion Chips Component for Data Exploration

This module provides UI components for displaying smart suggestion chips
that guide users toward deeper data exploration.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
import uuid
import json
from utils.smart_suggestions import get_dynamic_suggestions, handle_suggestion_click

def render_suggestion_chips(
    df: pd.DataFrame, 
    page_context: str,
    handle_action: Callable[[str, Dict[str, Any]], None],
    recent_actions: List[str] = None,
    max_chips: int = 5,
    container_key: str = None
):
    """
    Render smart suggestion chips for data exploration
    
    Args:
        df: Current pandas DataFrame
        page_context: Current page or section name
        handle_action: Callback function to handle chip clicks
        recent_actions: Optional list of recent user actions
        max_chips: Maximum number of chips to display
        container_key: Optional key for the container
    """
    # Generate a unique key if not provided
    if container_key is None:
        container_key = f"suggestion_container_{str(uuid.uuid4())[:8]}"
    
    # Get suggestions for current context
    suggestions = get_dynamic_suggestions(df, page_context, recent_actions)[:max_chips]
    
    # Create container with custom styling
    st.markdown("""
    <style>
    .suggestion-container {
        margin: 10px 0;
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render the suggestions
    with st.container():
        st.markdown("""
        <div class="suggestion-container">
        </div>
        """, unsafe_allow_html=True)
        
        # Empty state
        if not suggestions:
            st.markdown("*No suggestions available for current data*")
            return

        # Store suggestions in session state to access them when clicked
        suggestion_key = f"suggestions_{container_key}"
        st.session_state[suggestion_key] = suggestions
        
        # Create a horizontal layout
        cols = st.columns(min(len(suggestions), max_chips))
        
        # Display each suggestion as a chip
        for i, suggestion in enumerate(suggestions[:max_chips]):
            with cols[i]:
                chip_id = f"chip_{suggestion['id']}_{container_key}"
                
                # Add hover tooltip with description
                with st.container():
                    clicked = st.button(
                        suggestion['title'],
                        help=suggestion['description'],
                        key=chip_id
                    )
                    
                    if clicked:
                        # Call the handler with the suggestion action and parameters
                        handle_action(suggestion['action'], suggestion['parameters'])

def render_compact_suggestion_chips(
    df: pd.DataFrame,
    page_context: str,
    handle_action: Callable[[str, Dict[str, Any]], None],
    recent_actions: List[str] = None,
    container_key: str = None
):
    """
    Render a more compact version of suggestion chips for sidebars or small spaces
    
    Args:
        df: Current pandas DataFrame
        page_context: Current page or section name
        handle_action: Callback function to handle chip clicks
        recent_actions: Optional list of recent user actions
        container_key: Optional key for the container
    """
    # Generate a unique key if not provided
    if container_key is None:
        container_key = f"compact_suggestions_{str(uuid.uuid4())[:8]}"
    
    # Get suggestions for current context (limit to 3 for compact display)
    suggestions = get_dynamic_suggestions(df, page_context, recent_actions)[:3]
    
    # Store suggestions in session state to access them when clicked
    suggestion_key = f"suggestions_{container_key}"
    st.session_state[suggestion_key] = suggestions
    
    # Render section header
    st.markdown("### Suggested Actions")
    
    # Empty state
    if not suggestions:
        st.markdown("*No suggestions available*")
        return
    
    # Display each suggestion
    for suggestion in suggestions:
        chip_id = f"chip_{suggestion['id']}_{container_key}"
        
        # Create a button with the suggestion title
        clicked = st.button(
            suggestion['title'],
            help=suggestion['description'],
            key=chip_id
        )
        
        if clicked:
            # Call the handler with the suggestion action and parameters
            handle_action(suggestion['action'], suggestion['parameters'])

def handle_suggestion_action(action: str, parameters: Dict[str, Any]) -> bool:
    """
    Process a suggestion action
    
    Args:
        action: The action identifier
        parameters: Parameters for the action
        
    Returns:
        True if the action was handled, False otherwise
    """
    # Store the action in session state for later use
    if 'suggestion_actions' not in st.session_state:
        st.session_state.suggestion_actions = []
    
    # Add the action to history
    action_record = {
        'action': action,
        'parameters': parameters,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    st.session_state.suggestion_actions.append(action_record)
    
    # Set trigger for the specific action
    action_key = f"trigger_{action}"
    st.session_state[action_key] = parameters
    
    # Mark session state as needing a rerun
    st.session_state.suggestion_triggered = True
    
    return True

def check_triggered_action(action_name: str) -> Optional[Dict[str, Any]]:
    """
    Check if a specific action was triggered and get its parameters
    
    Args:
        action_name: Name of the action to check
        
    Returns:
        Parameters dict if the action was triggered, None otherwise
    """
    action_key = f"trigger_{action_name}"
    
    if action_key in st.session_state:
        # Get parameters and clear the trigger
        parameters = st.session_state[action_key]
        del st.session_state[action_key]
        return parameters
    
    return None

def create_category_chips(categories: List[str], active_category: str = None) -> str:
    """
    Create category selector chips and return the active category
    
    Args:
        categories: List of available categories
        active_category: Currently active category (if any)
        
    Returns:
        The currently selected category
    """
    st.write("Explore by category:")
    
    # Create horizontal layout
    cols = st.columns(len(categories))
    
    # Initialize return value
    selected = active_category if active_category else categories[0]
    
    # Create a button for each category
    for i, category in enumerate(categories):
        with cols[i]:
            is_active = category == active_category
            
            # Use different styling for active category
            if is_active:
                button_label = f"**{category}**"
            else:
                button_label = category
                
            if st.button(button_label, key=f"cat_{category}"):
                selected = category
    
    return selected

def create_exploration_flow(df: pd.DataFrame, initial_context: str = "general"):
    """
    Create a guided exploration flow using suggestion chips
    
    Args:
        df: The DataFrame to explore
        initial_context: Initial context for suggestions
    """
    # Initialize the exploration path in session state if not exists
    if 'exploration_path' not in st.session_state:
        st.session_state.exploration_path = [{
            'step': 0,
            'context': initial_context,
            'action': 'start',
            'parameters': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }]
    
    # Get current step
    current_step = len(st.session_state.exploration_path) - 1
    current_context = st.session_state.exploration_path[-1]['context']
    
    # Show current exploration path
    st.markdown("### Your Exploration Path")
    for i, step in enumerate(st.session_state.exploration_path):
        prefix = "→" if i == current_step else "✓"
        st.markdown(f"{prefix} **Step {i+1}**: {step['action'].replace('_', ' ').title()}")
    
    # Show suggestions for next steps
    st.markdown("### Suggested Next Steps")
    
    # Create handler for suggestion clicks
    def handle_next_step(action, parameters):
        # Add to exploration path
        st.session_state.exploration_path.append({
            'step': current_step + 1,
            'context': _get_next_context(action),
            'action': action,
            'parameters': parameters,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        # Trigger rerun to update UI
        st.experimental_rerun()
    
    # Render suggestion chips
    render_suggestion_chips(
        df=df,
        page_context=current_context,
        handle_action=handle_next_step,
        recent_actions=[step['action'] for step in st.session_state.exploration_path],
        container_key=f"flow_{current_step}"
    )
    
    # Render the content for the current step
    _render_exploration_step(df, current_step)
    
    # Option to reset exploration
    if st.button("Reset Exploration Path"):
        st.session_state.exploration_path = [{
            'step': 0,
            'context': initial_context,
            'action': 'start',
            'parameters': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }]
        st.experimental_rerun()

def _get_next_context(action: str) -> str:
    """Get next context based on action"""
    if action.startswith('show_'):
        return "exploration"
    elif action.startswith('create_'):
        return "visualization"
    elif action.startswith('perform_'):
        return "statistical"
    else:
        return "general"

def _render_exploration_step(df: pd.DataFrame, step_index: int):
    """Render the content for an exploration step"""
    if step_index <= 0:
        # First step is just the starting point
        st.markdown("Select a suggestion above to begin exploring your data.")
        return
    
    # Get the step information
    step = st.session_state.exploration_path[step_index]
    action = step['action']
    params = step['parameters']
    
    # Render based on action type
    st.markdown(f"### {action.replace('_', ' ').title()}")
    
    if action == 'show_summary_statistics':
        st.dataframe(df.describe())
        
    elif action == 'show_correlation_matrix':
        columns = params.get('columns', df.select_dtypes(include=['number']).columns.tolist())
        st.dataframe(df[columns].corr())
        
    elif action == 'group_by_analysis':
        group_col = params.get('group_column')
        agg_cols = params.get('aggregation_columns', [])
        
        if group_col and agg_cols:
            result = df.groupby(group_col)[agg_cols].agg(['mean', 'count'])
            st.dataframe(result)
        
    elif action == 'create_histogram':
        column = params.get('column')
        if column and column in df.columns:
            import plotly.express as px
            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)
        
    elif action == 'create_scatter_plot':
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            import plotly.express as px
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        
    elif action == 'create_box_plot':
        num_col = params.get('numeric_column')
        cat_col = params.get('categorical_column')
        
        if num_col and cat_col and num_col in df.columns and cat_col in df.columns:
            import plotly.express as px
            fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
        
    elif action == 'analyze_missing_values':
        # Show missing value counts by column
        missing = df.isnull().sum().to_frame('Missing Values')
        missing['Percentage'] = (df.isnull().sum() / len(df) * 100).round(2)
        st.dataframe(missing[missing['Missing Values'] > 0])
        
    elif action == 'detect_outliers':
        columns = params.get('columns', [])
        
        if columns:
            # Simple outlier detection using IQR
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    st.write(f"Outliers in {col}: {len(outliers)} records")
                    if len(outliers) > 0:
                        st.dataframe(outliers.head(10))
        
    elif action == 'perform_clustering':
        columns = params.get('columns', [])
        n_clusters = params.get('n_clusters', 3)
        
        if columns and all(col in df.columns for col in columns):
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import plotly.express as px
            
            # Prepare data
            X = df[columns].copy()
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_temp = df.copy()
            df_temp['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Show cluster distribution
            st.write("Cluster Distribution:")
            st.write(df_temp['cluster'].value_counts())
            
            # Visualize if 2 or more columns
            if len(columns) >= 2:
                fig = px.scatter(
                    df_temp, 
                    x=columns[0], 
                    y=columns[1], 
                    color='cluster',
                    title=f"Clusters based on {columns[0]} and {columns[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
    elif action == 'reduce_dimensions':
        columns = params.get('columns', [])
        n_components = params.get('n_components', 2)
        
        if columns and all(col in df.columns for col in columns):
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            import plotly.express as px
            
            # Prepare data
            X = df[columns].copy()
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X_scaled)
            
            # Create result dataframe
            df_pca = pd.DataFrame(
                components, 
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            # Display variance explained
            st.write("Variance Explained:")
            explained_var = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Variance Explained (%)': pca.explained_variance_ratio_ * 100
            })
            st.dataframe(explained_var)
            
            # Visualize if 2 components
            if n_components >= 2:
                fig = px.scatter(
                    df_pca, 
                    x='PC1', 
                    y='PC2',
                    title="PCA: First Two Principal Components"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif action == 'perform_regression':
        target_col = params.get('target_column')
        feature_cols = params.get('feature_columns', [])
        
        if target_col and feature_cols and target_col in df.columns and all(col in df.columns for col in feature_cols):
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score
            
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            
            # Display results
            st.write(f"R² Score: {r2:.4f}")
            
            # Show coefficients
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_
            })
            st.write("Feature Coefficients:")
            st.dataframe(coef_df.sort_values('Coefficient', ascending=False))
            
            # Plot actual vs predicted
            import plotly.express as px
            fig = px.scatter(
                x=y, 
                y=y_pred,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title=f"Actual vs Predicted {target_col}"
            )
            # Add diagonal line
            fig.add_shape(
                type="line",
                line=dict(dash="dash"),
                x0=y.min(),
                y0=y.min(),
                x1=y.max(),
                y1=y.max()
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.write(f"Action '{action}' not implemented yet.")