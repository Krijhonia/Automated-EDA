import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DataInsight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.target_column = None
        
    def load_data(self, uploaded_file):
        """Load and validate uploaded data"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return False
            
            # Identify column types
            self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def generate_summary_stats(self):
        """Generate comprehensive summary statistics"""
        if self.data is None:
            return None
        
        summary = {
            'basic_info': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'numeric_columns': len(self.numeric_columns),
                'categorical_columns': len(self.categorical_columns),
                'missing_values': self.data.isnull().sum().sum(),
                'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
            },
            'numeric_summary': self.data.describe() if self.numeric_columns else pd.DataFrame(),
            'missing_analysis': self.data.isnull().sum().sort_values(ascending=False),
            'data_types': self.data.dtypes
        }
        
        return summary
    
    def detect_outliers(self, column):
        """Detect outliers using IQR method"""
        if column not in self.numeric_columns:
            return []
        
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers
    
    def perform_correlation_analysis(self):
        """Perform correlation analysis on numeric columns"""
        if not self.numeric_columns:
            return None
        
        correlation_matrix = self.data[self.numeric_columns].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'matrix': correlation_matrix,
            'high_correlations': high_corr_pairs
        }
    
    def perform_clustering(self, n_clusters=3):
        """Perform K-means clustering on numeric data"""
        if len(self.numeric_columns) < 2:
            return None
        
        # Prepare data
        cluster_data = self.data[self.numeric_columns].dropna()
        if len(cluster_data) == 0:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to original data
        result_data = cluster_data.copy()
        result_data['Cluster'] = cluster_labels
        
        return {
            'data': result_data,
            'centers': kmeans.cluster_centers_,
            'labels': cluster_labels,
            'inertia': kmeans.inertia_
        }
    
    def build_predictive_model(self, target_col, model_type='auto'):
        """Build and evaluate predictive models"""
        if target_col not in self.data.columns:
            return None
        
        # Prepare features
        feature_cols = [col for col in self.numeric_columns if col != target_col]
        if not feature_cols:
            return None
        
        # Clean data
        model_data = self.data[feature_cols + [target_col]].dropna()
        if len(model_data) < 50:  # Minimum samples for meaningful model
            return None
        
        X = model_data[feature_cols]
        y = model_data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Determine model type
        if model_type == 'auto':
            if y.dtype == 'object' or len(y.unique()) < 10:
                model_type = 'classification'
            else:
                model_type = 'regression'
        
        # Train model
        if model_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'type': 'classification',
                'model': model,
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actual': y_test,
                'classification_report': classification_report(y_test, y_pred)
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'type': 'regression',
                'model': model,
                'r2_score': r2,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actual': y_test
            }

def create_visualizations(analyzer):
    """Create various visualizations"""
    if analyzer.data is None:
        return
    
    # Distribution plots for numeric columns
    if analyzer.numeric_columns:
        st.subheader("📈 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_num_col = st.selectbox("Select numeric column for distribution", analyzer.numeric_columns)
            
            fig = px.histogram(
                analyzer.data, 
                x=selected_num_col,
                nbins=30,
                title=f"Distribution of {selected_num_col}",
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(analyzer.numeric_columns) > 1:
                selected_num_col2 = st.selectbox("Select second numeric column", [col for col in analyzer.numeric_columns if col != selected_num_col])
                
                fig = px.scatter(
                    analyzer.data,
                    x=selected_num_col,
                    y=selected_num_col2,
                    title=f"{selected_num_col} vs {selected_num_col2}",
                    color_discrete_sequence=['#764ba2']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Categorical analysis
    if analyzer.categorical_columns:
        st.subheader("📊 Categorical Analysis")
        
        selected_cat_col = st.selectbox("Select categorical column", analyzer.categorical_columns)
        
        # Value counts
        value_counts = analyzer.data[selected_cat_col].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Top 10 values in {selected_cat_col}",
                color=value_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {selected_cat_col}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 DataInsight Pro</h1>
        <p>Advanced Automated Data Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your dataset", 
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Load Data"):
                with st.spinner("Loading data..."):
                    if analyzer.load_data(uploaded_file):
                        st.success("Data loaded successfully!")
                        st.session_state.data_loaded = True
                    else:
                        st.error("Failed to load data")
        
        if analyzer.data is not None:
            st.markdown("### 📋 Analysis Options")
            analysis_options = st.multiselect(
                "Select analysis types:",
                ["Summary Statistics", "Correlation Analysis", "Outlier Detection", 
                 "Clustering", "Predictive Modeling", "Advanced Visualizations"],
                default=["Summary Statistics"]
            )
    
    # Main content
    if analyzer.data is not None:
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(analyzer.data.head(100), use_container_width=True)
        
        # Summary statistics
        if "Summary Statistics" in analysis_options:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.subheader("📊 Summary Statistics")
            
            summary = analyzer.generate_summary_stats()
            
            # Basic info metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{summary['basic_info']['rows']:,}")
            with col2:
                st.metric("Total Columns", summary['basic_info']['columns'])
            with col3:
                st.metric("Numeric Columns", summary['basic_info']['numeric_columns'])
            with col4:
                st.metric("Missing Values", f"{summary['basic_info']['missing_values']:,}")
            
            # Detailed statistics
            if not summary['numeric_summary'].empty:
                st.subheader("Numeric Column Statistics")
                st.dataframe(summary['numeric_summary'], use_container_width=True)
            
            # Missing values analysis
            if summary['missing_analysis'].sum() > 0:
                st.subheader("Missing Values Analysis")
                missing_df = pd.DataFrame({
                    'Column': summary['missing_analysis'].index,
                    'Missing Count': summary['missing_analysis'].values,
                    'Missing Percentage': (summary['missing_analysis'].values / len(analyzer.data) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Correlation Analysis
        if "Correlation Analysis" in analysis_options and analyzer.numeric_columns:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.subheader("🔗 Correlation Analysis")
            
            corr_analysis = analyzer.perform_correlation_analysis()
            
            if corr_analysis:
                # Correlation heatmap
                fig = px.imshow(
                    corr_analysis['matrix'],
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title="Correlation Matrix"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlations
                if corr_analysis['high_correlations']:
                    st.subheader("High Correlations (|r| > 0.7)")
                    high_corr_df = pd.DataFrame(corr_analysis['high_correlations'])
                    st.dataframe(high_corr_df, use_container_width=True)
                else:
                    st.info("No high correlations found (|r| > 0.7)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Outlier Detection
        if "Outlier Detection" in analysis_options and analyzer.numeric_columns:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.subheader("🎯 Outlier Detection")
            
            outlier_col = st.selectbox("Select column for outlier detection", analyzer.numeric_columns)
            outliers = analyzer.detect_outliers(outlier_col)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Outliers Found", len(outliers))
            with col2:
                st.metric("Outlier Percentage", f"{len(outliers)/len(analyzer.data)*100:.2f}%")
            
            if len(outliers) > 0:
                # Box plot
                fig = px.box(analyzer.data, y=outlier_col, title=f"Box Plot - {outlier_col}")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Outlier details
                st.subheader("Outlier Details")
                st.dataframe(outliers, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clustering Analysis
        if "Clustering" in analysis_options and len(analyzer.numeric_columns) >= 2:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.subheader("🎨 Clustering Analysis")
            
            n_clusters = st.slider("Number of clusters", 2, 8, 3)
            
            if st.button("Perform Clustering"):
                with st.spinner("Performing clustering..."):
                    clustering_result = analyzer.perform_clustering(n_clusters)
                
                if clustering_result:
                    # Cluster visualization
                    if len(analyzer.numeric_columns) >= 2:
                        fig = px.scatter(
                            clustering_result['data'],
                            x=analyzer.numeric_columns[0],
                            y=analyzer.numeric_columns[1],
                            color='Cluster',
                            title=f"K-means Clustering ({n_clusters} clusters)",
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2c3e50')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = clustering_result['data'].groupby('Cluster').agg(['mean', 'count']).round(2)
                    st.dataframe(cluster_stats, use_container_width=True)
                    
                    st.info(f"Inertia (within-cluster sum of squares): {clustering_result['inertia']:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predictive Modeling
        if "Predictive Modeling" in analysis_options and analyzer.numeric_columns:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.subheader("🤖 Predictive Modeling")
            
            target_col = st.selectbox("Select target variable", analyzer.numeric_columns)
            
            if st.button("Build Model"):
                with st.spinner("Building predictive model..."):
                    model_result = analyzer.build_predictive_model(target_col)
                
                if model_result:
                    # Model performance
                    col1, col2 = st.columns(2)
                    with col1:
                        if model_result['type'] == 'classification':
                            st.metric("Model Accuracy", f"{model_result['accuracy']:.3f}")
                        else:
                            st.metric("R² Score", f"{model_result['r2_score']:.3f}")
                    
                    with col2:
                        st.metric("Model Type", model_result['type'].title())
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    fig = px.bar(
                        model_result['feature_importance'],
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance",
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2c3e50')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Predictions vs Actual
                    st.subheader("Predictions vs Actual")
                    comparison_df = pd.DataFrame({
                        'Actual': model_result['actual'],
                        'Predicted': model_result['predictions']
                    })
                    
                    fig = px.scatter(
                        comparison_df,
                        x='Actual',
                        y='Predicted',
                        title="Predictions vs Actual Values",
                        color_discrete_sequence=['#667eea']
                    )
                    # Add diagonal line
                    min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
                    max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
                    fig.add_shape(
                        type="line",
                        x0=min_val, y0=min_val,
                        x1=max_val, y1=max_val,
                        line=dict(dash="dash", color="red")
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2c3e50')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not build model. Please check your data.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced Visualizations
        if "Advanced Visualizations" in analysis_options:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.subheader("📊 Advanced Visualizations")
            create_visualizations(analyzer)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("💡 Key Insights")
        
        insights = []
        summary = analyzer.generate_summary_stats()
        
        # Data quality insights
        missing_pct = (summary['basic_info']['missing_values'] / (summary['basic_info']['rows'] * summary['basic_info']['columns'])) * 100
        if missing_pct > 10:
            insights.append(f"⚠️ High missing data rate: {missing_pct:.1f}% of values are missing")
        elif missing_pct > 0:
            insights.append(f"✅ Low missing data rate: {missing_pct:.1f}% of values are missing")
        else:
            insights.append("✅ No missing values detected")
        
        # Data size insights
        if summary['basic_info']['rows'] > 10000:
            insights.append(f"📈 Large dataset: {summary['basic_info']['rows']:,} rows available for analysis")
        elif summary['basic_info']['rows'] < 100:
            insights.append(f"⚠️ Small dataset: Only {summary['basic_info']['rows']:,} rows - consider collecting more data")
        
        # Column type insights
        if summary['basic_info']['numeric_columns'] > summary['basic_info']['categorical_columns']:
            insights.append("🔢 Numeric-heavy dataset - well-suited for statistical analysis and modeling")
        elif summary['basic_info']['categorical_columns'] > summary['basic_info']['numeric_columns']:
            insights.append("📝 Categorical-heavy dataset - consider encoding for machine learning")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="analysis-section">
            <h2>🚀 Welcome to DataInsight Pro</h2>
            <p>Your comprehensive automated data analysis platform. Upload your dataset to get started with:</p>
            <ul>
                <li>📊 Comprehensive summary statistics</li>
                <li>🔗 Correlation analysis</li>
                <li>🎯 Outlier detection</li>
                <li>🎨 K-means clustering</li>
                <li>🤖 Predictive modeling</li>
                <li>📈 Advanced visualizations</li>
                <li>💡 Automated insights</li>
            </ul>
            <p>Simply upload your CSV or Excel file using the sidebar to begin your analysis journey!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()