"""
Portuguese Address Search - Streamlit Web Interface
=================================================

This is a simple web interface for testing and demonstrating the Portuguese
address search baseline system. It provides an intuitive way to interact with
the hybrid PostGIS + Elasticsearch architecture.

Features:
- Real-time search with fuzzy matching and typo tolerance
- Results with confidence scores and GPS coordinates
- Search performance metrics display
- Interactive map visualization (optional)
- Examples demonstrating different query types

Usage:
    streamlit run search_ui.py
    
Then navigate to http://localhost:8501 in your browser
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import List
import folium
from streamlit_folium import folium_static

# Import our search API
from src.baseline_framework.search_api import HybridAddressSearch, SearchResult

# Configure Streamlit page
st.set_page_config(
    page_title="Portuguese Address Search",
    page_icon="🇵🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .search-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .metric-container {
        background-color: #e9ecef;
        padding: 0.8rem;
        border-radius: 0.3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_search_system():
    """Initialize the search system (cached to avoid reloading)"""
    try:
        with st.spinner("🔄 Initializing Portuguese Address Search System..."):
            search_system = HybridAddressSearch()
            return search_system
    except Exception as e:
        st.error(f"❌ Failed to initialize search system: {e}")
        return None

def format_confidence_score(score: float) -> str:
    """Format confidence score with color coding"""
    if score >= 50:
        return f'<span class="confidence-high">{score:.1f}%</span>'
    elif score >= 20:
        return f'<span class="confidence-medium">{score:.1f}%</span>'
    else:
        return f'<span class="confidence-low">{score:.1f}%</span>'

def display_search_result(result: SearchResult, index: int):
    """Display a single search result with formatted information"""
    
    # Main address display
    st.markdown(f"""
    <div class="search-result">
        <h4>🏠 {result.address_full}</h4>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>Confidence:</strong> {format_confidence_score(result.confidence_score)}<br>
                <strong>ES Score:</strong> {result.elasticsearch_score:.2f}
            </div>
            <div style="text-align: right;">
                <strong>Location:</strong><br>
                {result.latitude:.6f}, {result.longitude:.6f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional details in expander
    with st.expander(f"📋 Details for result #{index + 1}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Address Components:**")
            st.write(f"• Street: {result.street_clean or 'N/A'}")
            st.write(f"• House Number: {result.housenumber_primary or 'N/A'}")
            if result.housenumber_specifier:
                st.write(f"• Specifier: {result.housenumber_specifier}")
            st.write(f"• City: {result.city_clean or 'N/A'}")
            st.write(f"• Postcode: {result.postcode_clean or 'N/A'}")
        
        with col2:
            st.write("**Geographic Information:**")
            st.write(f"• Municipality: {result.municipality or 'N/A'}")
            st.write(f"• District: {result.district or 'N/A'}")
            st.write(f"• OSM ID: {result.osm_id}")
            st.write(f"• OSM Type: {result.osm_type}")
            
            # Google Maps link
            if result.latitude and result.longitude:
                maps_url = f"https://www.google.com/maps?q={result.latitude},{result.longitude}"
                st.markdown(f"🗺️ [View on Google Maps]({maps_url})")

def create_results_map(results: List[SearchResult]) -> folium.Map:
    """Create a Folium map with search results"""
    if not results or not any(r.latitude and r.longitude for r in results):
        return None
    
    # Calculate map center
    valid_coords = [(r.latitude, r.longitude) for r in results if r.latitude and r.longitude]
    if not valid_coords:
        return None
    
    center_lat = sum(lat for lat, lon in valid_coords) / len(valid_coords)
    center_lon = sum(lon for lat, lon in valid_coords) / len(valid_coords)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add markers for each result
    for i, result in enumerate(results):
        if result.latitude and result.longitude:
            confidence_color = 'green' if result.confidence_score >= 50 else 'orange' if result.confidence_score >= 20 else 'red'
            
            folium.Marker(
                [result.latitude, result.longitude],
                popup=f"""
                <b>{result.address_full}</b><br>
                Confidence: {result.confidence_score:.1f}%<br>
                ES Score: {result.elasticsearch_score:.2f}
                """,
                tooltip=f"#{i+1}: {result.address_full}",
                icon=folium.Icon(color=confidence_color, icon='home')
            ).add_to(m)
    
    return m

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.title("🇵🇹 Portuguese Address Search System")
    st.markdown("### Hybrid PostGIS + Elasticsearch Architecture Baseline")
    
    # Sidebar with information and settings
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This search system demonstrates the baseline implementation of the 
        Portuguese address search using a hybrid architecture:
        
        **🔍 Search Features:**
        • Fuzzy matching with typo tolerance
        • Abbreviation expansion (r. → rua)
        • Postcode-only searches
        • Multi-field matching
        • Geographic coordinate lookup
        
        **⚡ Performance Metrics:**
        Based on benchmark analysis showing significant improvements 
        for complex queries over naive approaches.
        """)
        
        st.header("🛠️ Settings")
        max_results = st.slider("Max Results", min_value=1, max_value=20, value=10)
        min_score = st.slider("Min Confidence Threshold", min_value=0.0, max_value=10.0, value=0.1, step=0.1)
        show_map = st.checkbox("Show Results Map", value=True)
        include_raw = st.checkbox("Include Raw Data", value=False)
        
        st.header("📝 Example Queries")
        st.markdown("""
        Try these example searches:
        • `rua augusta lisboa`
        • `av liberdade 123`
        • `1000-001`
        • `r santa catarina porto`
        • `coimbra`
        • `rua agusta, lisbon` (typo test)
        """)
    
    # Initialize search system
    search_system = initialize_search_system()
    
    if search_system is None:
        st.error("❌ Cannot initialize search system. Please check your database connections.")
        st.stop()
    
    # Display system status
    st.success("✅ Search system initialized successfully!")
    
    # Get system stats
    stats = search_system.get_search_stats()
    if stats['total_searches'] > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Searches", stats['total_searches'])
        with col2:
            st.metric("Avg ES Time", f"{stats['avg_elasticsearch_time']:.3f}s")
        with col3:
            st.metric("Avg PostGIS Time", f"{stats['avg_postgis_time']:.3f}s")
        with col4:
            st.metric("Avg Total Time", f"{stats['avg_total_time']:.3f}s")
    
    st.markdown("---")
    
    # Main search interface
    st.header("🔍 Address Search")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., rua augusta lisboa, av liberdade 123, or 1000-001",
        help="You can search by street name, full address, or postcode. Abbreviations and typos are handled automatically."
    )
    
    # Search button and logic
    if st.button("🔍 Search", type="primary") or query:
        if query:
            try:
                # Perform search with timing
                search_start = time.time()
                
                with st.spinner(f"🔄 Searching for '{query}'..."):
                    results = search_system.search(
                        query=query,
                        max_results=max_results,
                        min_score=min_score,
                        include_raw=include_raw
                    )
                
                search_time = time.time() - search_start
                
                # Display results
                if results:
                    st.success(f"✅ Found {len(results)} results in {search_time:.3f} seconds")
                    
                    # Results summary
                    avg_confidence = sum(r.confidence_score for r in results) / len(results)
                    best_result = max(results, key=lambda r: r.confidence_score)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Results Found", len(results))
                    with col2:
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    with col3:
                        st.metric("Best Match", f"{best_result.confidence_score:.1f}%")
                    
                    # Display results
                    st.markdown("### 📍 Search Results")
                    
                    for i, result in enumerate(results):
                        display_search_result(result, i)
                    
                    # Optional map display
                    if show_map:
                        st.markdown("### 🗺️ Results Map")
                        results_map = create_results_map(results)
                        if results_map:
                            folium_static(results_map, width=700, height=500)
                        else:
                            st.info("No valid coordinates available for map display.")
                    
                    # Export results option
                    if st.checkbox("📊 Export Results as CSV"):
                        df_data = []
                        for result in results:
                            df_data.append({
                                'Address': result.address_full,
                                'Street': result.street_clean,
                                'City': result.city_clean,
                                'Postcode': result.postcode_clean,
                                'Municipality': result.municipality,
                                'District': result.district,
                                'Latitude': result.latitude,
                                'Longitude': result.longitude,
                                'Confidence_Score': result.confidence_score,
                                'ES_Score': result.elasticsearch_score
                            })
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning(f"⚠️ No results found for '{query}'. Try adjusting your search terms or reducing the minimum confidence threshold.")
                    
                    # Suggestions for improving search
                    st.markdown("""
                    **💡 Search Tips:**
                    - Try using abbreviations: `r.` for `rua`, `av` for `avenida`
                    - Search by postcode only: `1000-001`
                    - Use partial addresses: `augusta lisboa`
                    - Check for typos in your query
                    """)
            
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                st.exception(e)
        
        else:
            st.warning("⚠️ Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Portuguese Address Search System | Hybrid Architecture Baseline<br>
        Built with Streamlit, PostGIS, and Elasticsearch</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()