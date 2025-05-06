import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import altair as alt
import re
import os
import json

# Configure threading for Numba to avoid the workqueue threading layer issue
# Use a threading layer that doesn't require external dependencies
os.environ["NUMBA_THREADING_LAYER"] = "omp"
# Alternatively, can also try these if issues persist:
# os.environ["NUMBA_THREADING_LAYER"] = "forksafe"
# os.environ["NUMBA_THREADING_LAYER"] = "threadsafe"

# Try to import UMAP if available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    st.sidebar.warning("UMAP not installed. Consider installing it with: pip install umap-learn")

# Try to import plotly, but handle it gracefully if there's an error
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except (ImportError, AttributeError) as e:
    st.warning("Plotly not available or encountering an error. Will use Matplotlib for 3D visualization instead.")
    PLOTLY_AVAILABLE = False

# Set page title and layout
st.set_page_config(page_title="Criminal Offense Embedding Visualization", layout="wide")
st.title("Criminal Offense Embedding Visualization")

# Load the SentenceTransformer model
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model

# Load and process the exclusions data
@st.cache_data
def load_exclusions_data():
    with open("exclusions.txt", "r") as file:
        lines = file.readlines()
    
    # Filter out empty lines and strip whitespace
    offenses = [line.strip() for line in lines if line.strip()]
    
    # Create a dataframe
    df = pd.DataFrame({
        'offense': offenses,
        'type': 'non-reportable'
    })
    
    return df

# Generate embeddings
@st.cache_data
def generate_embeddings(texts, _model):
    embeddings = _model.encode(texts)
    return embeddings

# Augment text with context
def augment_with_context(offense, offense_type):
    if offense_type == 'non-reportable':
        return offense
    elif offense_type == 'felony':
        return f"Serious criminal felony offense: {offense}. This is a crime punishable by imprisonment for more than one year or death."
    elif offense_type == 'custom_offense':
        # Don't augment custom offenses - let them be classified naturally
        return offense
    return offense

# Apply dimensionality reduction
def apply_dimension_reduction(embeddings, method='tsne', perplexity=30, n_components=2, n_neighbors=15, min_dist=0.1):
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    elif method == 'umap' and UMAP_AVAILABLE:
        # Configure UMAP to use a single thread to avoid threading issues
        reducer = UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            random_state=42,
            n_jobs=1  # Force single-threaded execution
        )
    else:  # PCA
        reducer = PCA(n_components=n_components)
    
    reduced_data = reducer.fit_transform(embeddings)
    return reduced_data

# Calculate similarity between felonies and non-reportable offenses
def calculate_similarity_stats(embeddings, types):
    felony_indices = [i for i, t in enumerate(types) if t == 'felony']
    non_reportable_indices = [i for i, t in enumerate(types) if t == 'non-reportable']
    
    if not felony_indices or not non_reportable_indices:
        return None
    
    felony_embeddings = embeddings[felony_indices]
    non_reportable_embeddings = embeddings[non_reportable_indices]
    
    # Calculate cosine similarity between each felony and all non-reportable offenses
    similarities = []
    for felony_embedding in felony_embeddings:
        # Reshape to 2D arrays for cosine_similarity
        felony_2d = felony_embedding.reshape(1, -1)
        sim_scores = cosine_similarity(felony_2d, non_reportable_embeddings).flatten()
        similarities.append({
            'mean': np.mean(sim_scores),
            'median': np.median(sim_scores),
            'min': np.min(sim_scores),
            'max': np.max(sim_scores),
            'std': np.std(sim_scores)
        })
    
    return similarities

# Function to create a Matplotlib 3D visualization
def create_matplotlib_3d_plot(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Check if 'category' column exists
    if 'category' in df.columns:
        # Get unique categories
        categories = df['category'].unique()
        
        # Create a colormap
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Plot each category
        for i, category in enumerate(categories):
            mask = df['category'] == category
            if mask.any():
                color = colors[i % len(colors)]
                ax.scatter(
                    df.loc[mask, 'x'],
                    df.loc[mask, 'y'],
                    df.loc[mask, 'z'],
                    c=color,
                    label=category,
                    alpha=0.7,
                    s=30
                )
    else:
        # Fallback to original logic
        for offense_type, color, label in [
            ('non-reportable', 'blue', 'Non-reportable offense'),
            ('felony', 'red', 'Felony offense'), 
            ('custom_offense', 'orange', 'Custom felony offense')
        ]:
            mask = df['type'] == offense_type
            if mask.any():
                ax.scatter(
                    df.loc[mask, 'x'],
                    df.loc[mask, 'y'],
                    df.loc[mask, 'z'],
                    c=color,
                    label=label,
                    alpha=0.7,
                    s=30
                )
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Visualization of Offenses')
    ax.legend()
    
    return fig

# Load felony clusters from JSON file
@st.cache_data
def load_felony_clusters():
    try:
        with open("embedding-diverse-criminal-records.json", "r") as file:
            clusters = json.load(file)
        return clusters
    except Exception as e:
        st.error(f"Error loading felony clusters: {e}")
        return {}

# Load exclusions data
df_exclusions = load_exclusions_data()

# Load felony clusters
felony_clusters = load_felony_clusters()

# Create list of felony categories from clusters
felony_categories = list(felony_clusters.keys()) if felony_clusters else []

# Sample felony offenses for the demo (fallback if JSON not available)
sample_felonies = [
    "Armed Robbery",
    "Aggravated Assault",
    "Burglary",
    "Drug Trafficking",
    "Identity Theft",
    "Kidnapping",
    "Murder",
    "Sexual Assault",
    "Grand Larceny",
    "Fraud"
]

# Add sidebar
st.sidebar.header("Settings")

# Add model selection
model_name = st.sidebar.selectbox(
    "Embedding Model",
    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"],
    index=0
)

# Load model based on selection
model = load_model(model_name)

# Add approach selection
approach = st.sidebar.selectbox(
    "Approach",
    ["Basic Embedding", "Augmented Context"],
    index=0  # Keep Basic Embedding as default
)

reduction_method = st.sidebar.selectbox(
    "Dimension Reduction Method",
    ["UMAP", "t-SNE", "PCA"] if UMAP_AVAILABLE else ["t-SNE", "PCA"],
    index=2  # Set PCA as default (index 2)
)

if reduction_method == "t-SNE":
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
elif reduction_method == "UMAP" and UMAP_AVAILABLE:
    n_neighbors = st.sidebar.slider("n_neighbors", 5, 100, 50)  # Default to 50
    min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.03)    # Default to 0.15
else:
    # For PCA
    perplexity = 30
    n_neighbors = 50   # Updated default
    min_dist = 0.15    # Updated default

# Add visualization dimension option
visualization_dim = st.sidebar.selectbox(
    "Visualization Dimensions",
    ["2D", "3D"],
    index=1  # Default to 2D
)

# User input for custom felony offense
st.sidebar.header("Enter Custom Offense")
custom_felony = st.sidebar.text_input("Custom Offense (will be automatically classified)")

# Select felony categories
st.sidebar.header("Select Felony Categories")
selected_categories = st.sidebar.multiselect(
    "Select felony categories to include",
    felony_categories if felony_categories else sample_felonies,
    default=["Armed Robbery", "Murder", "Burglary"] if not felony_categories else felony_categories[:3]
)

# Option to select specific examples for each category
show_examples = st.sidebar.checkbox("Select specific examples for each category", value=False)

# Option to include/exclude non-reportable offenses
include_nonreportable = st.sidebar.checkbox("Include non-reportable offenses", value=True)

selected_examples = {}
if show_examples and felony_clusters:
    for category in selected_categories:
        st.sidebar.subheader(f"{category} Examples")
        if category in felony_clusters:
            examples = felony_clusters[category]
            selected = st.sidebar.multiselect(
                f"Select examples for {category}",
                examples,
                default=examples[:2]  # Default to first 2 examples
            )
            selected_examples[category] = selected

# Create a combined dataframe with both non-reportable offenses and felonies
def prepare_data():
    # Get a sample of non-reportable offenses (optional, to reduce computational load)
    sample_size = min(1000, len(df_exclusions))
    df_non_reportable = df_exclusions.sample(sample_size, random_state=42)
    
    # Add the selected felonies
    felonies = []
    
    if felony_clusters and selected_categories:
        # Use the detailed felony clusters from JSON
        for category in selected_categories:
            if show_examples and category in selected_examples:
                # Add only selected examples for this category
                for example in selected_examples[category]:
                    felonies.append({
                        'offense': example, 
                        'type': 'felony', 
                        'category': category
                    })
            else:
                # Add all examples for this category
                if category in felony_clusters:
                    for example in felony_clusters[category]:
                        felonies.append({
                            'offense': example, 
                            'type': 'felony', 
                            'category': category
                        })
    else:
        # Fallback to simple felony categories as before
        for felony in selected_categories:
            felonies.append({'offense': felony, 'type': 'felony', 'category': felony})
        
    # Add the custom felony if provided
    if custom_felony:
        felonies.append({'offense': custom_felony, 'type': 'custom_offense', 'category': 'Custom'})
    
    df_felonies = pd.DataFrame(felonies)
    
    # Combine both datasets - conditionally include non-reportable offenses
    if include_nonreportable:
        df_combined = pd.concat([df_non_reportable, df_felonies], ignore_index=True)
    else:
        df_combined = df_felonies.copy()
    
    # Ensure 'category' column exists for all rows
    if 'category' not in df_combined.columns:
        df_combined['category'] = np.nan
    df_combined['category'] = df_combined['category'].fillna(df_combined['type'])
    
    return df_combined

# Main visualization
def visualize_data():
    # Get combined data
    df_combined = prepare_data()
    
    # List to store the raw offenses and augmented offenses
    raw_offenses = df_combined['offense'].tolist()
    types = df_combined['type'].tolist()
    
    # Process the data based on the selected approach
    if approach == "Augmented Context":
        st.info("Using Augmented Context approach: Adding context to felony offenses")
        processed_offenses = [augment_with_context(offense, type_) 
                             for offense, type_ in zip(raw_offenses, types)]
    else:
        processed_offenses = raw_offenses
    
    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = generate_embeddings(processed_offenses, model)
    
    # Apply dimensionality reduction
    n_components = 3 if visualization_dim == "3D" else 2
    
    with st.spinner(f"Applying {reduction_method}..."):
        if reduction_method == "UMAP" and UMAP_AVAILABLE:
            reduced_data = apply_dimension_reduction(
                embeddings, method='umap', n_components=n_components,
                n_neighbors=n_neighbors, min_dist=min_dist
            )
        else:
            method = 'tsne' if reduction_method == 't-SNE' else 'pca'
            reduced_data = apply_dimension_reduction(
                embeddings, method=method, perplexity=perplexity, n_components=n_components
            )
    
    # Add reduced dimensions to dataframe
    df_combined['x'] = reduced_data[:, 0]
    df_combined['y'] = reduced_data[:, 1]
    if n_components == 3:
        df_combined['z'] = reduced_data[:, 2]
    
    # Calculate similarity statistics
    similarity_stats = calculate_similarity_stats(embeddings, types)
    
    # Create visualization
    st.subheader(f"{visualization_dim} Visualization of Offenses using {reduction_method}")
    st.write(f"Model: {model_name}, Approach: {approach}")
    
    # Colors for different offense types
    colors = {
        'non-reportable': '#1f77b4',  # Blue
        'felony': '#d62728',          # Red
        'custom_offense': '#2ca02c'   # Green
    }
    
    if visualization_dim == "3D":
        if PLOTLY_AVAILABLE:
            try:
                # Create 3D visualization with Plotly
                color_column = 'category' if 'category' in df_combined.columns else 'type'
                
                fig = px.scatter_3d(
                    df_combined, 
                    x='x', y='y', z='z',
                    color=color_column,
                    hover_name='offense',
                    title=f"3D {reduction_method} Visualization of Offenses",
                    opacity=0.7,
                    width=800, 
                    height=600
                )
                
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(legend_title_text='Offense Category')
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error with Plotly: {e}")
                st.info("Falling back to Matplotlib 3D visualization")
                fig = create_matplotlib_3d_plot(df_combined)
                st.pyplot(fig)
        else:
            # Use matplotlib for 3D visualization
            fig = create_matplotlib_3d_plot(df_combined)
            st.pyplot(fig)
    else:
        # Create 2D visualization with Altair
        # Use category for color if available, otherwise use type
        color_column = 'category' if 'category' in df_combined.columns else 'type'
        
        # Get unique categories for color mapping
        categories = df_combined[color_column].unique()
        
        # Create a color scale with different colors for each category
        # Non-reportable stays blue, custom stays green, felony categories get different colors
        domain = list(categories)
        range_colors = ['#1f77b4']  # Start with blue for non-reportable
        
        # Add specific colors for felony categories
        category_colors = {
            'Armed Robbery': '#d62728',   # Red
            'Murder': '#ff7f0e',          # Orange
            'Burglary': '#9467bd',        # Purple
            'Drug Trafficking': '#8c564b', # Brown
            'Identity Theft': '#e377c2',   # Pink
            'Kidnapping': '#7f7f7f',       # Gray
            'Sexual Assault': '#bcbd22',   # Olive
            'Grand Larceny': '#17becf',    # Cyan
            'Fraud': '#2ca02c',           # Green
            'Aggravated Assault': '#d62728', # Red
            'custom_offense': '#2ca02c',   # Green for custom offense
            'Custom': '#2ca02c',           # Green for custom
            'non-reportable': '#1f77b4'    # Blue for non-reportable
        }
        
        # Add colors for each category
        for cat in domain:
            if cat == 'non-reportable':
                continue  # Already added blue
            elif cat in category_colors:
                range_colors.append(category_colors[cat])
            else:
                # For any other categories, assign a default color
                range_colors.append('#d62728')  # Default to red for other felonies
        
        chart = alt.Chart(df_combined).mark_circle(size=100).encode(
            x='x:Q',
            y='y:Q',
            color=alt.Color(f'{color_column}:N', 
                           scale=alt.Scale(domain=domain, range=range_colors)),
            tooltip=['offense', color_column]
        ).properties(
            width=800,
            height=500
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    
    # Display similarity analysis
    if similarity_stats:
        st.subheader("Similarity Analysis")
        st.write("This analysis shows how similar felony offenses are to non-reportable offenses based on their embeddings.")
        
        # Create a dataframe for the similarity stats
        felony_names = [offense for offense, type_ in zip(raw_offenses, types) if type_ == 'felony']
        df_similarity = pd.DataFrame(similarity_stats)
        df_similarity['offense'] = felony_names
        
        # Display the similarity stats
        st.write("Average similarity scores between each felony and non-reportable offenses (higher means more similar):")
        
        # Format the dataframe for display
        df_display = df_similarity[['offense', 'mean', 'min', 'max', 'std']].copy()
        df_display['mean'] = df_display['mean'].round(3)
        df_display['min'] = df_display['min'].round(3)
        df_display['max'] = df_display['max'].round(3)
        df_display['std'] = df_display['std'].round(3)
        df_display.columns = ['Felony Offense', 'Mean Similarity', 'Min Similarity', 'Max Similarity', 'Std Dev']
        
        st.table(df_display)
        
        # Plot the distribution of similarities
        st.write("Distribution of similarity scores:")
        
        # Create a figure with subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # For each felony, plot a histogram of its similarity to all non-reportable offenses
        for i, felony in enumerate(felony_names):
            felony_embedding = embeddings[[j for j, t in enumerate(types) if types[j] == 'felony'][i]]
            non_reportable_embeddings = embeddings[[j for j, t in enumerate(types) if t == 'non-reportable']]
            
            felony_2d = felony_embedding.reshape(1, -1)
            sim_scores = cosine_similarity(felony_2d, non_reportable_embeddings).flatten()
            
            # Plot histogram
            sns.kdeplot(sim_scores, label=felony, ax=ax)
        
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Similarities Between Felonies and Non-reportable Offenses')
        ax.legend()
        
        st.pyplot(fig)
    
    # Display statistics and comparisons
    st.subheader("Data Analysis")
    
    # Count of each type
    st.write("Count of offense types:")
    st.write(df_combined['type'].value_counts())
    
    # Analyze the differences between offense types
    if 'felony' in df_combined['type'].values or 'custom_offense' in df_combined['type'].values:
        st.write("### Offense Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("#### Non-reportable Offenses (sample):")
            non_reportable_sample = df_combined[df_combined['type'] == 'non-reportable'].sample(min(5, len(df_combined[df_combined['type'] == 'non-reportable'])))
            for offense in non_reportable_sample['offense']:
                st.write(f"- {offense}")
        
        with col2:
            st.write("#### Felony Offenses:")
            felony_offenses = df_combined[df_combined['type'] == 'felony']
            for offense in felony_offenses['offense']:
                st.write(f"- {offense}")
                
        with col3:
            custom_offenses = df_combined[df_combined['type'] == 'custom_offense']
            if not custom_offenses.empty:
                st.write("#### Custom Offense:")
                for offense in custom_offenses['offense']:
                    st.write(f"- {offense}")
                    
        # Add explanation about the different approaches
        st.write("""
        ### Approach Comparison
        
        #### Basic Embedding
        Simply embeds each offense using the sentence transformer model without any additional processing.
        
        #### Augmented Context
        Adds context to felony offenses to better distinguish them from non-reportable offenses. 
        This approach emphasizes that felonies are serious crimes with significant penalties.
        """)
        
        # Add explanation about model selection
        st.write("""
        ### Model Comparison
        
        #### all-MiniLM-L6-v2
        A smaller, faster model with 384-dimensional embeddings. Good for quick experiments but may not 
        capture as many nuances as larger models.
        
        #### all-mpnet-base-v2
        A larger model with 768-dimensional embeddings. Generally provides better semantic understanding 
        but requires more computational resources.
        
        #### all-distilroberta-v1
        Based on RoBERTa architecture. May capture different semantic aspects than the other models.
        """)
        
        # Add explanation about dimensionality reduction methods
        st.write("""
        ### Dimensionality Reduction Methods
        
        #### t-SNE
        Good at preserving local structure. The perplexity parameter controls the balance between preserving 
        local and global structure. Higher values preserve more global structure.
        
        #### PCA
        Preserves global variance but may not capture non-linear relationships between points.
        
        #### UMAP
        Often provides a better balance between preserving local and global structure than t-SNE. 
        The n_neighbors parameter controls how much global structure is preserved, while min_dist 
        affects how tightly points are packed together.
        """)

    # If a custom offense was provided, analyze which cluster it's closer to
    if 'custom_offense' in df_combined['type'].values:
        st.subheader("Custom Offense Analysis")
        
        # Get the custom offense and its embedding
        custom_idx = df_combined[df_combined['type'] == 'custom_offense'].index[0]
        custom_offense_text = df_combined.loc[custom_idx, 'offense']
        custom_embedding = embeddings[custom_idx].reshape(1, -1)
        
        # Get felony and non-reportable embeddings
        felony_indices = [i for i, t in enumerate(types) if t == 'felony']
        non_reportable_indices = [i for i, t in enumerate(types) if t == 'non-reportable']
        
        # Calculate average similarity to each group
        if felony_indices:
            felony_embeddings = embeddings[felony_indices]
            sim_to_felonies = np.mean(cosine_similarity(custom_embedding, felony_embeddings).flatten())
        else:
            sim_to_felonies = None
            
        sim_to_non_reportable = np.mean(cosine_similarity(custom_embedding, embeddings[non_reportable_indices]).flatten())
        
        # Display results
        st.write(f"Custom offense: **{custom_offense_text}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Similarity to Non-reportable Offenses", f"{sim_to_non_reportable:.3f}")
        with col2:
            if sim_to_felonies is not None:
                st.metric("Similarity to Felony Offenses", f"{sim_to_felonies:.3f}")
            else:
                st.write("No felony offenses selected for comparison")
        
        # Make a classification prediction
        if sim_to_felonies is not None:
            if sim_to_felonies > sim_to_non_reportable:
                prediction = "felony"
                confidence = (sim_to_felonies - sim_to_non_reportable) / (sim_to_felonies + sim_to_non_reportable)
            else:
                prediction = "non-reportable"
                confidence = (sim_to_non_reportable - sim_to_felonies) / (sim_to_felonies + sim_to_non_reportable)
            
            st.write(f"Based on embedding similarity, this offense is more likely to be a **{prediction}** (confidence: {confidence:.1%}).")
            st.write("Note: This is only a similarity-based prediction and should not be considered a legal classification.")

# Run the app
visualize_data() 
