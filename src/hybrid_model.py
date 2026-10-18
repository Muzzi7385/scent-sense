import re
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_perfumes
from src.preprocessing import preprocess_text
from src.fragrance_features import estimate_season, estimate_sweetness
from src.query_constraints import extract_constraints, apply_constraints

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
VIBE_PROFILES={
"seductive":["woody","spicy","amber","musk","sweet"],
"masculine":["woody","spicy","leather","oud","musk"],
"mysterious":["woody","smoky","amber","oud","spicy"],
"elegant":["floral","woody","musk","amber"],
"romantic":["floral","sweet","vanilla","musk"],
"fresh":["fresh","citrus","green","aquatic"],
"clean":["fresh","aquatic","citrus","musk"],
"energetic":["citrus","fresh","spicy","fruity"]}
SUGGESTED_PROMPTS=[
("🌙 Date night","I have a date tonight and want something seductive, masculine and memorable."),
("🍋 Fresh & clean","I want a fresh, clean fragrance with citrus and green notes for everyday use."),
("🖤 Mysterious","I want a mysterious fragrance with woody, smoky and spicy characteristics."),
("🌹 Romantic","I want a romantic fragrance with floral, vanilla and musky characteristics."),
("☀️ Summer","I want a refreshing fragrance for hot summer days with citrus and aquatic characteristics."),
("✨ Elegant","I want an elegant fragrance with woody, amber and musky characteristics.")]
FEATURES={
"fresh":["fresh","clean","crisp"],"woody":["woody","wood","cedarwood","sandalwood","oud","agarwood"],
"citrus":["citrus","lemon","bergamot","grapefruit","orange"],"green":["green","grassy","herbal"],
"floral":["floral","flower","rose","jasmine"],"sweet":["sweet","gourmand","vanilla","caramel","honey"],
"spicy":["spicy","pepper","cinnamon","nutmeg","saffron"],"aquatic":["aquatic","marine","oceanic"],
"fruity":["fruity","fruit","berry","apple","peach"],"smoky":["smoky","smoke","incense"],
"vanilla":["vanilla","vanillin"],"musk":["musk","musky","muskiness"],"amber":["amber","ambery"],
"leather":["leather","leathery"],"oud":["oud","agarwood"]}
def expand_vibe_query(query):
    q=str(query).lower(); additions=[]
    for vibe,features in VIBE_PROFILES.items():
        if re.search(rf"\b{re.escape(vibe)}\b",q): additions.extend(features)
    additions=list(dict.fromkeys(additions))
    return str(query)+(" "+" ".join(additions) if additions else "")
def calculate_feature_match(query,description):
    qt=set(re.findall(r"[a-z]+",str(query).lower())); dt=set(re.findall(r"[a-z]+",str(description).lower()))
    requested=[f for f,syns in FEATURES.items() if any(s in qt for s in syns)]
    if not requested: return 0.0,"No direct feature match"
    matched=[f for f in requested if any(s in dt for s in FEATURES[f])]
    return len(matched)/len(requested),(", ".join(matched) if matched else "No direct feature match")
@lru_cache(maxsize=1)
def _catalogue_tfidf():
    df=load_perfumes(); vec=TfidfVectorizer(ngram_range=(1,2),min_df=1)
    matrix=vec.fit_transform(df["search_text"].map(preprocess_text))
    return df,vec,matrix
@lru_cache(maxsize=1)
def _semantic():
    from sentence_transformers import SentenceTransformer
    df,_,_=_catalogue_tfidf()
    model=SentenceTransformer(MODEL_NAME)
    return model,model.encode(df["search_text"].tolist(),normalize_embeddings=True,show_progress_bar=False)
def recommend_hybrid(query,top_k=5,alpha=0.5):
    if not isinstance(query,str) or not query.strip(): raise ValueError("Please describe the fragrance you want.")
    if top_k<1: raise ValueError("top_k must be at least 1.")
    if not 0<=alpha<=1: raise ValueError("alpha must be between 0 and 1.")
    constraints=extract_constraints(query); expanded=expand_vibe_query(query)
    df,vec,matrix=_catalogue_tfidf()
    tf=cosine_similarity(vec.transform([preprocess_text(expanded)]),matrix).ravel()
    model,embeddings=_semantic()
    semantic=cosine_similarity(model.encode([expanded],normalize_embeddings=True),embeddings).ravel()
    def norm(a):
        a=np.asarray(a,dtype=float); span=a.max()-a.min() if len(a) else 0
        return (a-a.min())/span if span>0 else np.zeros_like(a)
    out=df.copy(); out["tfidf_score"]=tf; out["semantic_score"]=semantic
    out["hybrid_score"]=alpha*norm(tf)+(1-alpha)*norm(semantic)
    matches=out.apply(lambda r:calculate_feature_match(expanded,r.get("search_text",r.get("description",""))),axis=1)
    out["feature_match_score"]=matches.map(lambda x:x[0]); out["matched_features"]=matches.map(lambda x:x[1])
    out["season_estimate"]=out["description"].map(estimate_season)
    out["sweetness_estimate"]=out["description"].map(estimate_sweetness)
    out["final_score"]=.7*out["hybrid_score"]+.3*out["feature_match_score"]
    out=apply_constraints(out,constraints).sort_values("final_score",ascending=False)
    cols=["name","brand","description","price_inr","season_estimate","sweetness_estimate","tfidf_score","semantic_score","hybrid_score","feature_match_score","matched_features","final_score"]
    return out[[c for c in cols if c in out]].head(top_k).reset_index(drop=True)
