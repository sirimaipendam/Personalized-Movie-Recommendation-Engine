import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import re

# --- Data Loading and Preprocessing ---

def load_data(file_path):
    """Loads the movie dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print("Columns:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_genres(genres_str):
    """Parses comma-separated genre strings, cleans them, and returns a list."""
    if pd.isna(genres_str) or not isinstance(genres_str, str):
        return []
    # Split by comma, strip whitespace, and capitalize each genre
    return [genre.strip().capitalize() for genre in genres_str.split(',') if genre.strip()]

def create_movie_features(df):
    """Creates TF-IDF features for genres and overviews."""
    df['cleaned_genres'] = df['Genre'].apply(parse_genres)
    df['genres_str'] = df['cleaned_genres'].apply(lambda x: ' '.join(x))

    # Combine 'genres_str' and 'Overview' for TF-IDF vectorization
    df['combined_features'] = df['genres_str'].fillna('') + ' ' + df['Overview'].fillna('')

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10) # n_init to suppress warning
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    # Calculate cosine similarity for recommendations
    cosine_sim = cosine_similarity(tfidf_matrix)

    return df, kmeans, cosine_sim

# --- Recommendation Logic ---

def get_recommendations(selected_genre, num_recommendations, movies_df, kmeans_model, similarity_matrix):
    """Generates movie recommendations based on a selected genre."""
    print(f"[get_recommendations] Called with genre: {selected_genre}")
    # Clean the selected genre for consistent comparison
    cleaned_selected_genre = selected_genre.strip().capitalize()

    # Filter movies that contain the selected genre
    genre_filtered_movies = movies_df[movies_df['cleaned_genres'].apply(lambda x: cleaned_selected_genre in x)]

    if genre_filtered_movies.empty:
        print(f"[get_recommendations] No movies found for genre: {cleaned_selected_genre}")
        return None
    print(f"[get_recommendations] Found {len(genre_filtered_movies)} movies for genre: {cleaned_selected_genre}")

    # For simplicity, let's pick a random movie from the filtered list to get its cluster
    # In a real system, you might ask the user for a movie they liked or use a more sophisticated approach
    reference_movie = genre_filtered_movies.sample(1).iloc[0]
    reference_cluster = reference_movie['cluster']
    print(f"[get_recommendations] Reference movie: {reference_movie['Title']}, Cluster: {reference_cluster}")

    # Get movies from the same cluster
    cluster_movies = movies_df[movies_df['cluster'] == reference_cluster]
    print(f"[get_recommendations] Found {len(cluster_movies)} movies in the same cluster.")

    # Calculate similarity scores for movies within the cluster to the reference movie
    # Find the index of the reference movie in the original DataFrame
    ref_idx = movies_df[movies_df['Title'] == reference_movie['Title']].index[0]
    print(f"[get_recommendations] Reference movie index: {ref_idx}")

    # Get similarity scores for the reference movie against all movies in the original dataset
    sim_scores = list(enumerate(similarity_matrix[ref_idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar movies (excluding the reference movie itself)
    # Filter out movies that don't belong to the selected genre or are the reference movie
    recommended_indices = []
    for i, score in sim_scores:
        if i != ref_idx and cleaned_selected_genre in movies_df.iloc[i]['cleaned_genres']:
            recommended_indices.append(i)
        if len(recommended_indices) >= num_recommendations:
            break
    print(f"[get_recommendations] Number of recommended indices found: {len(recommended_indices)}")

    if not recommended_indices:
        print(f"[get_recommendations] No recommendations found in the same cluster for genre: {cleaned_selected_genre}")
        return None

    return movies_df.iloc[recommended_indices]

# --- GUI Components ---

class MovieCard(ttk.Frame):
    def __init__(self, parent, movie_data):
        super().__init__(parent, relief='solid', borderwidth=1, padding=10)
        self.columnconfigure(0, weight=1)

        title_label = ttk.Label(self, text=movie_data['Title'], font=('Helvetica', 14, 'bold'))
        title_label.grid(row=0, column=0, sticky='w', pady=(0, 5))

        # Display Rating
        rating_label = ttk.Label(self, text=f"Rating: {movie_data['Rating']}", font=('Helvetica', 10))
        rating_label.grid(row=1, column=0, sticky='w')

        # Removed the 'Genre' label as it's not explicitly requested
        # genre_label = ttk.Label(self, text=f"Genre: {', '.join(movie_data['cleaned_genres'])}", font=('Helvetica', 10))
        # genre_label.grid(row=2, column=0, sticky='w')

        overview_text = scrolledtext.ScrolledText(self, wrap='word', height=4, font=('Helvetica', 10))
        overview_text.insert(tk.END, movie_data['Overview'])
        overview_text.config(state='disabled') # Make it read-only
        overview_text.grid(row=3, column=0, sticky='nsew', pady=(5, 0))

class MovieRecommendationApp:
    def __init__(self, root, movies_df, kmeans, similarity_matrix):
        self.root = root
        self.root.title("Telugu Movie Recommender")
        self.movies_df = movies_df
        self.kmeans = kmeans
        self.similarity_matrix = similarity_matrix

        self.num_recommendations_var = tk.IntVar(self.root)
        self.num_recommendations_var.set(5) # Default to 5 recommendations

        self.create_widgets()

    def create_widgets(self):
        # Configure styles for debugging backgrounds
        style = ttk.Style()
        # Only apply style to ttk.Frame, not tk.Canvas
        style.configure("Lightblue.TFrame", background="lightblue")

        # Genre Selection
        genre_frame = ttk.Frame(self.root, padding="10")
        genre_frame.pack(pady=10)

        ttk.Label(genre_frame, text="Select a Genre:", font=("Helvetica", 12)).pack(side='left', padx=(0, 10))

        # Get unique genres from the dataset
        all_genres = sorted(list(set(g for genres_list in self.movies_df['cleaned_genres'] for g in genres_list)))
        self.genre_var = tk.StringVar(self.root)
        self.genre_var.set(all_genres[0] if all_genres else "") # Set default

        genre_option_menu = ttk.OptionMenu(genre_frame, self.genre_var, self.genre_var.get(), *all_genres)
        genre_option_menu.pack(side='left', padx=(0, 10))

        # Add a Spinbox for number of recommendations
        ttk.Label(genre_frame, text="Number of Movies:", font=("Helvetica", 12)).pack(side='left', padx=(10, 5))
        num_recommendations_spinbox = ttk.Spinbox(genre_frame, from_=1, to=20, textvariable=self.num_recommendations_var, width=5)
        num_recommendations_spinbox.pack(side='left')

        ttk.Button(genre_frame, text="Get Recommendations", command=self.show_recommendations).pack(side='left', padx=(10,0))

        # Results Display Area
        self.results_frame = ttk.Frame(self.root)
        self.results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.results_frame, bg='lightgray') # Reverted to bg for tk.Canvas
        self.canvas.pack(side='left', fill='both', expand=True)

        self.scrollbar = ttk.Scrollbar(self.results_frame, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side='right', fill='y')

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        # Bind the canvas to the scrollregion update and mouse wheel
        self.canvas.bind('<Configure>', self._on_canvas_configure) # Changed to a method
        self.canvas.bind('<MouseWheel>', lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.inner_frame = ttk.Frame(self.canvas, style="Lightblue.TFrame") # Applied style
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw', tags="self.inner_frame_tag")

        # Bind inner_frame to resize the canvas window
        self.inner_frame.bind("<Configure>", self._on_inner_frame_configure) # Changed to a method

    def _on_canvas_configure(self, event):
        # Resize the inner_frame window to match the canvas width
        self.canvas.itemconfig("self.inner_frame_tag", width=event.width)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_inner_frame_configure(self, event):
        # Update the scrollregion when the inner_frame size changes
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def show_recommendations(self):
        print("[show_recommendations] Button clicked.")
        genre = self.genre_var.get()
        num_recommendations = self.num_recommendations_var.get() # Get the selected number of recommendations

        if not genre:
            messagebox.showwarning("Warning", "Please select a genre")
            return
        print(f"[show_recommendations] Selected genre: {genre}, Number of recommendations: {num_recommendations}")

        # Clear previous recommendations and any existing loading indicator/title
        for widget in self.inner_frame.winfo_children():
            widget.destroy()

        # Loading indicator
        loading_frame = ttk.Frame(self.inner_frame)
        loading_frame.pack(pady=30)
        
        loading_label = ttk.Label(loading_frame,
                                 text="Finding recommendations",
                                 font=("Helvetica", 12))
        loading_label.pack()
        
        dots = ["."]*3
        self.after_id = None # Initialize after_id

        def animate_dots():
            nonlocal dots
            if loading_label.winfo_exists():
                dots = dots[1:] + dots[:1]
                loading_label.config(text=f"Finding recommendations{''.join(dots)}")
                self.after_id = self.root.after(500, animate_dots) # Store the after_id
            
        animate_dots()
        self.root.update_idletasks() # Update GUI to show loading

        # Fetch recommendations using the user-selected number
        recommended_movies = get_recommendations(
            genre, num_recommendations, self.movies_df, self.kmeans, self.similarity_matrix
        )
        print(f"[show_recommendations] Recommendations returned: {recommended_movies is not None and not recommended_movies.empty}")

        if self.after_id: # Check if an after call was scheduled
            self.root.after_cancel(self.after_id) # Cancel the scheduled call

        if loading_label.winfo_exists():
            loading_label.destroy()
        if loading_frame.winfo_exists(): # Ensure the loading frame is also destroyed
            loading_frame.destroy()

        if recommended_movies is not None and not recommended_movies.empty:
            # Display the title with the actual number of recommendations
            title_label = ttk.Label(self.inner_frame,
                                   text=f"Top {len(recommended_movies)} {genre} Movies",
                                   font=("Helvetica", 16, "bold"))
            title_label.pack(pady=20)

            for _, movie in recommended_movies.iterrows():
                card = MovieCard(self.inner_frame, movie)
                card.pack(pady=10, padx=20, fill='x', expand=True)
        else:
            error_label = ttk.Label(self.inner_frame,
                                   text=f"No recommendations found for {genre}",
                                   foreground="red") # Using a simple color for error
            error_label.pack(pady=20)

        # Update the scroll region after adding new widgets
        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Scroll to top
        self.canvas.yview_moveto(0)

# --- Main Execution ---

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'TeluguMovies_dataset.csv')
    movies_df = load_data(dataset_path)

    if movies_df is not None:
        # Rename columns to match expected names, ensuring 'Rating' is included
        movies_df = movies_df.rename(columns={'Genre': 'Genre', 'Overview': 'Overview', 'Movie': 'Title', 'Rating': 'Rating'})

        # Drop rows where 'Genre' or 'Overview' might be missing after renaming
        movies_df.dropna(subset=['Genre', 'Overview'], inplace=True)

        movies_df, kmeans_model, similarity_matrix = create_movie_features(movies_df)

        root = tk.Tk()
        app = MovieRecommendationApp(root, movies_df, kmeans_model, similarity_matrix)
        root.mainloop()
    else:
        print("Exiting: Could not load movie data.")
        