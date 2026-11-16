"""Generates treatments.json (basic, general advice - not a substitute for an agronomist)."""
import json, pathlib

H = ("Plant looks healthy. No treatment needed.",
     "Water at the base, keep good spacing and airflow, rotate crops, and scout weekly.")
T = {
 "Apple___Apple_scab": ("Remove fallen leaves and infected fruit. Spray a protectant fungicide (captan/mancozeb or sulfur) at bud break and repeat per label.", "Rake and destroy leaf litter in autumn; prune for airflow; plant resistant varieties."),
 "Apple___Black_rot": ("Prune out cankers and mummified fruit. Apply a labelled fungicide during the growing season.", "Sanitise pruning tools; remove dead wood; avoid wounding trees."),
 "Apple___Cedar_apple_rust": ("Apply a fungicide (myclobutanil or sulfur) from pink bud through petal fall.", "Remove nearby juniper/cedar hosts if possible; choose rust-resistant varieties."),
 "Apple___healthy": H,
 "Blueberry___healthy": H,
 "Cherry_(including_sour)___Powdery_mildew": ("Spray sulfur or a labelled fungicide; prune infected shoots.", "Improve canopy airflow; avoid excess nitrogen; irrigate in the morning."),
 "Cherry_(including_sour)___healthy": H,
 "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Apply a foliar fungicide if disease reaches the upper leaves before tasseling.", "Rotate crops, till residue, and plant resistant hybrids."),
 "Corn_(maize)___Common_rust_": ("Usually minor; apply a fungicide only if pustules are widespread on young plants.", "Plant resistant hybrids and sow early."),
 "Corn_(maize)___Northern_Leaf_Blight": ("Apply a foliar fungicide at early symptoms on susceptible hybrids.", "Rotate crops, bury residue, use resistant hybrids."),
 "Corn_(maize)___healthy": H,
 "Grape___Black_rot": ("Remove mummified berries and infected leaves; spray mancozeb or myclobutanil from early shoot growth.", "Open the canopy by pruning; destroy infected debris."),
 "Grape___Esca_(Black_Measles)": ("No cure. Remove and burn severely affected vines/wood; protect pruning wounds with sealant.", "Prune in dry weather, disinfect tools, avoid large pruning wounds."),
 "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Spray a copper-based or labelled fungicide; remove infected leaves.", "Improve airflow and avoid overhead irrigation."),
 "Grape___healthy": H,
 "Orange___Haunglongbing_(Citrus_greening)": ("No cure. Remove infected trees and control the Asian citrus psyllid with approved insecticides. Report to your local agriculture office.", "Plant certified disease-free nursery stock; monitor and control psyllids."),
 "Peach___Bacterial_spot": ("Apply copper sprays in early spring and a labelled bactericide during the season.", "Plant tolerant varieties; avoid overhead watering; avoid excess nitrogen."),
 "Peach___healthy": H,
 "Pepper,_bell___Bacterial_spot": ("Spray copper-based bactericide; remove badly infected plants.", "Use disease-free seed, rotate with non-solanaceous crops, avoid working in wet fields."),
 "Pepper,_bell___healthy": H,
 "Potato___Early_blight": ("Apply chlorothalonil or mancozeb at first symptoms; remove infected lower leaves.", "Rotate crops for 2-3 years; avoid plant stress; mulch."),
 "Potato___Late_blight": ("Act fast: remove infected plants and apply a labelled fungicide (e.g. metalaxyl/mancozeb or copper). Do not compost infected material.", "Use certified seed tubers, hill soil well, avoid overhead irrigation."),
 "Potato___healthy": H,
 "Raspberry___healthy": H,
 "Soybean___healthy": H,
 "Squash___Powdery_mildew": ("Spray sulfur, neem oil or potassium bicarbonate; remove heavily infected leaves.", "Space plants, plant in sun, choose resistant varieties."),
 "Strawberry___Leaf_scorch": ("Remove infected leaves; apply a labelled fungicide (e.g. captan) if severe.", "Use clean plants, renovate beds after harvest, avoid overhead watering."),
 "Strawberry___healthy": H,
 "Tomato___Bacterial_spot": ("Spray copper-based bactericide; remove infected leaves.", "Use clean seed, rotate crops, avoid handling wet plants."),
 "Tomato___Early_blight": ("Remove affected lower leaves; apply chlorothalonil, mancozeb or copper fungicide.", "Mulch, stake plants, rotate crops, water at the base."),
 "Tomato___Late_blight": ("Remove and destroy infected plants; apply a labelled fungicide (copper or chlorothalonil) immediately.", "Avoid leaf wetness; use resistant varieties; improve ventilation."),
 "Tomato___Leaf_Mold": ("Reduce humidity and increase ventilation; apply a labelled fungicide (chlorothalonil or copper).", "Prune lower leaves, avoid overhead watering, use resistant varieties."),
 "Tomato___Septoria_leaf_spot": ("Remove infected leaves; apply chlorothalonil, mancozeb or copper fungicide.", "Rotate crops, mulch, avoid wetting foliage."),
 "Tomato___Spider_mites Two-spotted_spider_mite": ("Spray insecticidal soap, neem oil or a labelled miticide; wash leaves with water.", "Keep plants well watered; encourage natural predators."),
 "Tomato___Target_Spot": ("Apply a labelled fungicide (chlorothalonil/azoxystrobin); remove infected leaves.", "Improve airflow, rotate crops, avoid overhead watering."),
 "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("No cure. Remove infected plants; control whiteflies with sticky traps, neem oil or approved insecticides.", "Use virus-resistant varieties, reflective mulch, and insect-proof nets in nurseries."),
 "Tomato___Tomato_mosaic_virus": ("No cure. Remove and destroy infected plants; disinfect hands and tools.", "Use certified seed, resistant varieties; do not smoke/handle tobacco near plants."),
 "Tomato___healthy": H,
}
out = {k: {"treatment": v[0], "prevention": v[1]} for k, v in T.items()}
for p in ("backend/treatments.json", "android/app/src/main/assets/treatments.json"):
    pathlib.Path(p).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(len(out), "classes written")
