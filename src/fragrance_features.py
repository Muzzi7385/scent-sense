import re
SUMMER = {"fresh","citrus","lemon","bergamot","grapefruit","aquatic","marine","green","mint","ozonic","light","neroli"}
WINTER = {"amber","vanilla","oud","smoky","smoke","leather","incense","spicy","cinnamon","tonka","gourmand","resin","warm","tobacco"}
SWEET = {"sweet","vanilla","caramel","honey","chocolate","gourmand","tonka","praline","marshmallow","sugar","benzoin"}

def _terms(text):
    return set(re.findall(r"[a-z]+", str(text).lower()))
def extract_accords(description):
    m = re.search(r"main\s*accords?\s*:\s*(.*?)(?=\|\s*(?:top|middle|base)\s*notes?\s*:|$)", str(description), re.I)
    return [x.strip().lower() for x in re.split(r"[,;/]",m.group(1)) if x.strip()] if m else []
def extract_notes(description):
    result=[]
    for section in ("top","middle","base"):
        m=re.search(rf"{section}\s*notes?\s*:\s*(.*?)(?=\|\s*(?:top|middle|base)\s*notes?\s*:|$)",str(description),re.I)
        if m: result.extend(x.strip().lower() for x in re.split(r"[,;/]",m.group(1)) if x.strip())
    return result
def estimate_season(description):
    terms=_terms(str(description)+" "+" ".join(extract_accords(description)+extract_notes(description)))
    s=len(terms & SUMMER); w=len(terms & WINTER)
    if s >= w+2: return "Summer"
    if w >= s+2: return "Winter"
    if s and not w: return "Spring/Summer"
    if w and not s: return "Autumn/Winter"
    return None
def estimate_sweetness(description):
    count=len(_terms(str(description)+" "+" ".join(extract_accords(description)+extract_notes(description))) & SWEET)
    return 4 if "sweet" in _terms(description) or count>=3 else 3 if count==2 else 2 if count==1 else None
