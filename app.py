import gradio as gr
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter("ignore")

# Load the model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load the dataset
ds = load_dataset("ada-datadruids/full_tmdb_movies_dataset")
movies = ds['train'].select(range(100000))

# Load the embeddings
all_embeddings = np.load('tmdb_embeddings.npy')

def is_english(idx):
    "A function to check if the movie is in English"
    return int(movies[int(idx)]['original_language'] == 'en')

def is_not_documentary(idx):
    "A function to check if the movie is not a documentary"
    movie = movies[int(idx)]
    if movie['genres'] is not None and 'documentry' in movie['genres'].lower():
        return 0
    if movie['overview'] is not None and 'documentry' in movie['overview'].lower():
        return 0
    return 1

def hybrid_score(similarity_score, popularity_score,alpha=0.5):
    "A function to calculate the hybrid score based on similarity and popularity scores"
    return alpha * popularity_score + (1 - alpha) * similarity_score

def check_min_rating(indices, min_rating):
    "A function to check if the movie rating is above the minimum rating"
    return [int(movies[int(idx)]['vote_average'] >= min_rating) for idx in indices]

def recommend(overview, keywords_list, is_english_filter, is_not_documentary_filter, min_rating, alpha, N):
    """
    Recommends movies based on content similarity and popularity, with optional filters.
    
    Args:
        overview (str): Movie plot description to use as search query
        keywords_list (list): List of keywords/tags to enhance search
        is_english_filter (bool): Whether to filter for English-language movies
        is_not_documentary_filter (bool): Whether to exclude documentary films
        min_rating (float): Minimum rating threshold for movies (0-10 scale)
        alpha (float): Weighting factor between content similarity (0) and popularity (1)
        N (int): Maximum number of recommendations to return
        
    Returns:
        tuple: (scores, indices) where:
            - scores (np.array): Array of recommendation scores
            - indices (np.array): Array of corresponding movie indices
    """
    
    # Combine overview and keywords into a single search query
    keywords = ' '.join(keywords_list)
    description = f'Overview: {overview}. Keywords: {keywords}.'
    
    # Encode the search query into embedding vector
    description_encoded = model.encode(description)
    description_encoded = description_encoded.reshape(1, -1)  # Reshape for cosine similarity
    
    # Calculate cosine similarity between query and all movies
    sim_matrix = cosine_similarity(all_embeddings, description_encoded)
    sim_scores = sim_matrix.flatten()
    
    # Get top 100 most similar movies (before filtering)
    top_indices = np.argsort(sim_scores)[-100:][::-1]
    top_scores = sim_scores[top_indices]
    
    # Initialize filters (default to passing all movies)
    english_filter = np.ones(top_indices.shape).astype(int)
    documentary_filter = np.ones(top_indices.shape).astype(int)
    
    # Apply language filter if enabled
    if is_english_filter:
        english_filter = np.array([is_english(idx) for idx in top_indices])
    
    # Apply documentary filter if enabled
    if is_not_documentary_filter:
        documentary_filter = np.array([is_not_documentary(idx) for idx in top_indices])
    
    # Apply minimum rating filter
    min_rating_filter = check_min_rating(top_indices, min_rating)
    
    # Combine all filters
    keep_mask = english_filter & documentary_filter & min_rating_filter
    top_indices = top_indices[keep_mask.astype(bool)] 
    top_scores = top_scores[keep_mask.astype(bool)]
    
    # Get popularity scores for remaining movies
    top_popularity = np.array(ds['train'][top_indices]['popularity'])
    
    # Normalize popularity to 0-1 range
    normalized_popularity = (top_popularity - top_popularity.min()) / (top_popularity.max() - top_popularity.min())
    
    # Calculate hybrid score combining similarity and popularity
    final_scores = alpha * normalized_popularity + (1 - alpha) * top_scores
    
    # Sort movies by final hybrid score
    sorted_indices = np.argsort(final_scores)[::-1]  # Descending order
    sorted_final_scores = final_scores[sorted_indices]
    sorted_final_indices = top_indices[sorted_indices]
    
    # Return top N recommendations (or all if fewer than N remain)
    if len(sorted_final_indices) >= N:
        return sorted_final_scores[:N], sorted_final_indices[:N]
    else:
        return sorted_final_scores, sorted_final_indices

def process_inputs(overview, keywords, english_filter, documentary_filter, min_rating, alpha):
    # Convert alpha from 0-100 to 0.0-1.0
    alpha = alpha / 100.0
    keywords_list = keywords.split(",")
    
    # Process the inputs through your recommendation system
    recommendations_score, recommendation_indices = recommend(
        overview=overview,
        keywords_list=keywords_list,
        is_english_filter=english_filter,
        is_not_documentary_filter=documentary_filter,
        min_rating=min_rating,
        alpha=alpha,
        N=3
    )
    
    # Start building HTML output
    html_output = """
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 800px;
        margin: 0 auto;
    ">
        <h2 style="color: #4f46e5; text-align: center; margin-bottom: 30px;">Top Recommendations</h2>
    """
    
    for i, idx in enumerate(recommendation_indices):
        movie = movies[int(idx)]
        year = movie['release_date'].split('-')[0] if movie['release_date'] else 'Unknown'
        
        html_output += f"""
        <div style="
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
            transition: transform 0.3s ease;
        ">
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            ">
                <div style="
                    background: linear-gradient(135deg, #3498db, #2c3e50);
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    margin-right: 15px;
                ">{i+1}</div>
                <h3 style="
                    margin: 0;
                    color: #2c3e50;
                ">{movie['title']} <span style="color: #7f8c8d;">({year})</span></h3>
            </div>
            
            <div style="display: flex; margin-bottom: 15px;">
                <div style="
                    background: #f1c40f;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-weight: bold;
                    display: inline-flex;
                    align-items: center;
                ">
                    <span style="margin-right: 5px;">⭐</span> {movie['vote_average']}/10
                </div>
                
                <div style="
                    background: #ecf0f1;
                    color: #2c3e50;
                    padding: 5px 10px;
                    border-radius: 20px;
                    margin-left: 10px;
                ">
                    {movie['genres']}
                </div>
            </div>
            
            <div style="
                background: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            ">
                <p style="margin: 0; color: #34495e;">{movie['overview']}</p>
            </div>
        </div>
        """
    
    html_output += "</div>"
    
    return gr.HTML(html_output)



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Movie Recommendation System
    *Discover your next favorite film based on your preferences*
    """)
    
    with gr.Row():
        with gr.Column():
            overview = gr.Textbox(label="Describe what you're looking for", 
                                lines=3,
                                placeholder="A space adventure with complex characters...")
            
            keywords = gr.Textbox(
                label="Keywords (comma separated)",
                placeholder="sci-fi, action, adventure"
            )
            
        with gr.Column():
            with gr.Group():
                english_filter = gr.Checkbox(value=True, label="English Only")
                documentary_filter = gr.Checkbox(value=True, label="Exclude Documentaries")

            min_rating = gr.Slider(1, 10, value=7, 
                            step=0.5,
                            label="Minimum Rating")
            
            alpha = gr.Slider(0, 100, value=80, 
                            label="Recommendation Balance",
                            info="Left: More relevant | Right: More popular")
            
            submit_btn = gr.Button("Find Movies", variant="primary")
        
    # Output
    # Replace this line in your interface:
# output = gr.Textbox(label="Recommendations", interactive=False)

# With this:
    output = gr.HTML(label="Recommendations")
    
    # Connect the function
    submit_btn.click(
        fn=process_inputs,
        inputs=[overview, keywords, english_filter, documentary_filter, min_rating, alpha],
        outputs=output
    )

demo.launch()
