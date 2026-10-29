"""
ml.py — scikit-learn genre classifier for Somnium
TF-IDF vectoriser + LinearSVC pipeline
Trained on a synthetic corpus of genre-labelled sentences at startup.
"""
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'genre_model.pkl')

# ── Synthetic training corpus ─────────────────────────────────────────────────
# Each entry: (text, genre)
_CORPUS = [
    # Fantasy
    ("I flew over an enchanted castle with dragons and a wizard casting spells", "Fantasy"),
    ("A mystical elf guided me through an ancient magical kingdom", "Fantasy"),
    ("I discovered a fairy village hidden beneath a glowing mushroom forest", "Fantasy"),
    ("A sorcerer turned the entire realm to gold with a flick of his wand", "Fantasy"),
    ("Unicorns galloped across the sky while pixies sang enchanted songs", "Fantasy"),
    ("The dragon breathed fire that transformed into magical butterflies", "Fantasy"),
    ("I wielded a sword of light to defeat an evil warlock in enchanted ruins", "Fantasy"),
    ("Potions bubbled in the wizard tower as spells glowed on parchment", "Fantasy"),

    # Horror
    ("A shadowy monster chased me through a dark abandoned house", "Horror"),
    ("Ghosts screamed in the crumbling mansion as blood dripped from the walls", "Horror"),
    ("I was trapped by a demon that wore a human face in the darkness", "Horror"),
    ("Zombies flooded the streets while I hid in a cellar full of corpses", "Horror"),
    ("A vampire watched from the shadows while sinister laughter echoed", "Horror"),
    ("Something evil lurked behind every door as I crept through the terror", "Horror"),
    ("A knife-wielding figure chased me through fog-filled streets at night", "Horror"),
    ("Dead figures rose from graves as I tried to scream but no sound came", "Horror"),

    # Adventure
    ("I explored a jungle island searching for buried treasure with a map", "Adventure"),
    ("My ship sailed across the ocean towards uncharted mountains", "Adventure"),
    ("I climbed a cliff face to discover ancient ruins hidden in the wilderness", "Adventure"),
    ("A quest took me through forests rivers and deserts to rescue someone", "Adventure"),
    ("I parachuted into a remote valley and had to survive alone", "Adventure"),
    ("Riding horses across vast plains we discovered a hidden civilisation", "Adventure"),
    ("An expedition through the caves revealed a lost underground world", "Adventure"),
    ("I navigated rapids on a raft fighting against the raging river", "Adventure"),

    # Mystery
    ("A secret door led to a hidden room full of mysterious clues", "Mystery"),
    ("I played detective investigating a puzzle with hidden suspects", "Mystery"),
    ("Someone was leaving cryptic ciphers around an unknown city", "Mystery"),
    ("The maze had no exit and every door revealed a new secret conspiracy", "Mystery"),
    ("I had to solve a riddle before the truth about the crime was revealed", "Mystery"),
    ("Whispers pointed to a suspect in the dark underground investigation", "Mystery"),
    ("The detective uncovered a hidden layer beneath the apparent crime scene", "Mystery"),
    ("Clues scattered across the museum pointed to an impossible conspiracy", "Mystery"),

    # Sci-Fi
    ("I travelled through space to a distant planet orbiting two stars", "Sci-Fi"),
    ("Robots marched across the neon city as AI took control of the future", "Sci-Fi"),
    ("An alien spacecraft landed and communicated through pulsing light", "Sci-Fi"),
    ("I stepped through a dimension portal into a cyberpunk world of chrome", "Sci-Fi"),
    ("The android asked if it had a soul while the galaxy burned outside", "Sci-Fi"),
    ("Lasers lit up the asteroid belt during an interstellar war", "Sci-Fi"),
    ("Teleporting between planets I witnessed technology that bent physics", "Sci-Fi"),
    ("A time machine took me to the far future where machines governed all", "Sci-Fi"),

    # Romance
    ("I danced under the stars with someone I deeply loved", "Romance"),
    ("A tender kiss at sunset made my heart overflow with passion", "Romance"),
    ("We shared a long embrace in a rose garden full of affection", "Romance"),
    ("My soulmate appeared in the dream and our hearts connected instantly", "Romance"),
    ("A wedding on a hilltop filled with warmth love and blossoming flowers", "Romance"),
    ("I longed for them and felt their presence warm beside me all night", "Romance"),
    ("Hand in hand we walked through a field of blooming roses at dusk", "Romance"),
    ("The feeling of being deeply cherished wrapped around me like warmth", "Romance"),

    # Thriller
    ("A spy mission went wrong and I was being hunted by assassins", "Thriller"),
    ("I uncovered a government conspiracy and had to run before capture", "Thriller"),
    ("Dodging snipers I carried classified information through a war zone", "Thriller"),
    ("An undercover operation turned deadly when the betrayal was revealed", "Thriller"),
    ("Explosives were planted everywhere and I had seconds to escape", "Thriller"),
    ("A mole in the agency had leaked my identity to enemy operatives", "Thriller"),
    ("Armed men chased my car through narrow streets at high speed", "Thriller"),
    ("I defused a bomb seconds before it would have destroyed the city", "Thriller"),

    # Surreal
    ("I floated through melting walls while colours morphed into sounds", "Surreal"),
    ("Everything shrank and expanded impossibly as I drifted through void", "Surreal"),
    ("Flying without wings I passed through layers of shifting dimensions", "Surreal"),
    ("The sky was liquid and I swam upward through infinite rainbow fog", "Surreal"),
    ("Objects morphed into other objects endlessly in a loop with no logic", "Surreal"),
    ("I watched myself from outside my body as reality distorted around me", "Surreal"),
    ("Gravity reversed and buildings became impossible stairs to nowhere", "Surreal"),
    ("A bizarre parade of giant talking clocks led me into a mirror world", "Surreal"),

    # Spiritual
    ("Angels descended in a beam of divine light filling me with peace", "Spiritual"),
    ("I meditated in a sacred temple and felt my soul leave my body", "Spiritual"),
    ("A spiritual guide led me to enlightenment in a cosmic realm of light", "Spiritual"),
    ("The cosmos spoke to me and I understood the karma of all existence", "Spiritual"),
    ("I transcended physical form and merged with a holy divine consciousness", "Spiritual"),
    ("Prayers echoed in a celestial space as sacred geometry surrounded me", "Spiritual"),
    ("My soul ascended through clouds toward a luminous divine presence", "Spiritual"),
    ("I heard the universe breathe and felt at peace with all living things", "Spiritual"),

    # Comedy
    ("A clown chased me with a giant rubber chicken through a silly maze", "Comedy"),
    ("Everything was absurdly funny and everyone laughed at bizarre pranks", "Comedy"),
    ("I accidentally tripped into a cartoon world full of hilarious nonsense", "Comedy"),
    ("A ridiculous series of mishaps turned a simple errand into chaos", "Comedy"),
    ("Everyone in the dream wore silly costumes at an absurd party", "Comedy"),
    ("I gave a speech and everything went comically wrong in front of crowds", "Comedy"),
    ("A dog drove a car while cats filed paperwork in a giggling office", "Comedy"),
    ("The most ridiculous situation unfolded and I could not stop laughing", "Comedy"),

    # Nightmare
    ("I tried to run but my legs would not move no matter how hard I tried", "Nightmare"),
    ("My teeth crumbled out one by one while I stared in the mirror in panic", "Nightmare"),
    ("I was falling from a great height with no way to stop or wake up", "Nightmare"),
    ("Something was chasing me down an endless hallway and every door was locked", "Nightmare"),
    ("I was drowning and could not reach the surface no matter how I struggled", "Nightmare"),
    ("I screamed for help but no sound came out of my mouth at all", "Nightmare"),
    ("I was buried alive and could feel the walls closing in around me", "Nightmare"),
    ("I woke up inside the dream and realised I was still trapped and could not escape", "Nightmare"),

    # Lucid
    ("I realised I was dreaming and chose to fly straight up into the clouds", "Lucid"),
    ("Once I knew it was a dream I changed the scenery just by wishing it", "Lucid"),
    ("I pinched myself, confirmed I was dreaming, and summoned a door to anywhere", "Lucid"),
    ("Aware this was a dream, I controlled the gravity and walked up the walls", "Lucid"),
    ("I noticed the odd details that proved I was dreaming and took control of the plot", "Lucid"),
    ("Knowing I was asleep, I called out and reshaped the dream world around me", "Lucid"),
    ("I stabilised the dream by rubbing my hands and then decided where to go next", "Lucid"),
    ("Recognising the dream signs, I stayed calm and steered the dream exactly where I wanted", "Lucid"),

    # Supernatural
    ("A ghost drifted through the locked door and stood silently at the foot of my bed", "Supernatural"),
    ("Spirits whispered from the walls of the old house as the lights flickered on their own", "Supernatural"),
    ("An unseen presence moved objects across the room while the temperature dropped suddenly", "Supernatural"),
    ("A poltergeist hurled furniture around the attic while I hid behind the door", "Supernatural"),
    ("The medium channelled a voice that was not her own as candles guttered out", "Supernatural"),
    ("Shadows peeled off the walls and moved independently across the ceiling", "Supernatural"),
    ("A haunted mirror showed a reflection that moved a second after I did", "Supernatural"),
    ("The seance table rattled and cold hands gripped mine in the dark", "Supernatural"),

    # Apocalyptic
    ("The sky turned red as meteors fell and cities crumbled into dust around me", "Apocalyptic"),
    ("Survivors scavenged through a ruined city after the collapse of civilisation", "Apocalyptic"),
    ("A plague swept the world and I wandered empty streets looking for other survivors", "Apocalyptic"),
    ("The ground split open and ash blotted out the sun as everyone fled", "Apocalyptic"),
    ("I hoarded supplies in a bunker while the world outside burned to the ground", "Apocalyptic"),
    ("Floodwaters swallowed the last standing buildings as the final broadcast went silent", "Apocalyptic"),
    ("Robots and ash filled the horizon in the last days of the old world", "Apocalyptic"),
    ("I led a small group of survivors through the wasteland searching for clean water", "Apocalyptic"),

    # War
    ("Artillery shook the trench as soldiers around me ducked for cover under fire", "War"),
    ("I carried a wounded comrade across no man's land while bullets cracked overhead", "War"),
    ("Tanks rolled through the ruined village as the resistance regrouped in the cellars", "War"),
    ("Orders came to hold the line as the enemy advanced through the smoke", "War"),
    ("I radioed for reinforcements while explosions lit up the night sky", "War"),
    ("The platoon marched through mud toward a front line that never seemed to end", "War"),
    ("A ceasefire was called just as the final assault was about to begin", "War"),
    ("I buried a fallen friend's dog tags beneath a tree near the battlefield", "War"),

    # Crime
    ("I planned a heist on a vault guarded by lasers and rotating cameras", "Crime"),
    ("The getaway car sped through red lights while sirens closed in behind us", "Crime"),
    ("A crime boss demanded the stolen diamonds be returned before midnight", "Crime"),
    ("I picked the lock of a pawn shop looking for the missing evidence", "Crime"),
    ("The gang split the loot in a warehouse while a rival crew watched from outside", "Crime"),
    ("I laundered the cash through a chain of fake businesses across the city", "Crime"),
    ("A informant tipped off the police just as the robbery was underway", "Crime"),
    ("I hid the gun under the floorboards before the detectives searched the house", "Crime"),

    # Western
    ("Dust rose over the dry plain as a lone rider approached the saloon at noon", "Western"),
    ("I drew my revolver as the outlaw's gang rode into the one-street town", "Western"),
    ("The sheriff posted a bounty on the train robbers hiding in the canyon", "Western"),
    ("Cattle stampeded across the ranch as cowboys scrambled to turn the herd", "Western"),
    ("A standoff at high noon left the whole town watching from behind shuttered windows", "Western"),
    ("I tracked the horse thief across the desert until the trail went cold", "Western"),
    ("The stagecoach was ambushed at the pass just before it reached the fort", "Western"),
    ("Tumbleweeds rolled past the saloon doors as the piano player fell silent", "Western"),

    # Historical
    ("I stood among togas and marble columns as senators debated in the ancient forum", "Historical"),
    ("Knights in full armour jousted before a medieval castle crowded with banners", "Historical"),
    ("I sailed on a wooden galleon charting unknown coastlines centuries ago", "Historical"),
    ("Pharaohs were carried through torch-lit halls lined with hieroglyphs and gold", "Historical"),
    ("I marched with Victorian factory workers through smoke-filled cobbled streets", "Historical"),
    ("A royal court whispered of conspiracy behind velvet curtains and candlelight", "Historical"),
    ("I witnessed a revolution unfold in the square as old banners were torn down", "Historical"),
    ("Monks copied manuscripts by candlelight in a stone abbey centuries in the past", "Historical"),

    # Drama
    ("My family argued around the dinner table about a secret that had been kept for years", "Drama"),
    ("I confronted my estranged sister about the inheritance that tore our family apart", "Drama"),
    ("A courtroom fell silent as the verdict about my brother was finally read aloud", "Drama"),
    ("I sat beside a hospital bed holding my parent's hand through the long night", "Drama"),
    ("The wedding speech turned into a tearful confession in front of every guest", "Drama"),
    ("I packed my bags while my partner begged me to reconsider leaving", "Drama"),
    ("An old letter revealed a truth about my childhood I had never been told", "Drama"),
    ("I stood at the funeral trying to find the words I never got to say", "Drama"),

    # Tragedy
    ("I watched helplessly as the ship I was on slowly sank beneath the waves", "Tragedy"),
    ("A fire swept through the theatre before anyone could reach the only exit", "Tragedy"),
    ("I arrived moments too late to say goodbye before the ambulance left", "Tragedy"),
    ("The bridge collapsed just as the last car was crossing in the rain", "Tragedy"),
    ("I buried my childhood home under rubble after the earthquake levelled the street", "Tragedy"),
    ("A single mistake cost us everything we had spent years building together", "Tragedy"),
    ("I read the telegram twice before the news truly sank in", "Tragedy"),
    ("The whole town gathered in silence as the flag was lowered to half mast", "Tragedy"),

    # Action
    ("I leapt from a moving train onto a motorcycle to escape the explosion behind me", "Action"),
    ("Gunfire erupted as I vaulted over rooftops chasing the van through the city", "Action"),
    ("I disarmed the bomb with seconds left while dodging falling debris", "Action"),
    ("A car chase tore through the market as crates and stalls flew into the street", "Action"),
    ("I fought off three attackers on a collapsing bridge above the river", "Action"),
    ("The helicopter dropped a rope just as the building's roof gave way beneath me", "Action"),
    ("I sprinted across rooftops under gunfire to reach the extraction point in time", "Action"),
    ("A fistfight spilled out of the bar and into the middle of a busy highway", "Action"),

    # Psychological
    ("I could not tell if the voice in my head belonged to me or someone else", "Psychological"),
    ("The same hallway kept repeating no matter which direction I turned", "Psychological"),
    ("I questioned whether the people around me were real or figments of my own mind", "Psychological"),
    ("A therapist in the dream kept asking questions that unravelled my own memories", "Psychological"),
    ("I woke up inside another dream and could no longer trust what was real", "Psychological"),
    ("My reflection spoke to me with thoughts I had never said out loud", "Psychological"),
    ("I obsessively recounted the same memory until it stopped making sense", "Psychological"),
    ("Paranoia crept in as everyone's faces seemed to shift when I looked away", "Psychological"),

    # Noir
    ("Rain streaked the window of my office as a mysterious client walked in at midnight", "Noir"),
    ("A smoky jazz bar hid secrets behind every half-lit booth and whispered deal", "Noir"),
    ("I followed the dame in the red coat through alleys lit only by neon signs", "Noir"),
    ("The detective's trench coat dripped rain as he lit a cigarette under a streetlamp", "Noir"),
    ("A double-crossing partner left me holding the blame for a crime I didn't commit", "Noir"),
    ("I pieced together the case from torn photographs scattered across my desk", "Noir"),
    ("The femme fatale's story didn't add up, but I followed the lead anyway", "Noir"),
    ("Venetian blinds cut the moonlight into stripes across the dead man's floor", "Noir"),

    # Domestic
    ("I was washing dishes while my family argued gently about weekend chores", "Domestic"),
    ("We rearranged the living room furniture while the kids napped upstairs", "Domestic"),
    ("I burned the dinner again and we ended up ordering takeout and laughing about it", "Domestic"),
    ("My parents and siblings gathered in the kitchen making an ordinary Sunday breakfast", "Domestic"),
    ("I folded laundry while the television played quietly in the next room", "Domestic"),
    ("We painted the nursery together, arguing lightly over which shade of yellow", "Domestic"),
    ("I was fixing a leaky faucet while my partner read on the couch nearby", "Domestic"),
    ("The whole family squeezed around a small table for an ordinary weeknight dinner", "Domestic"),

    # Travel
    ("I missed my connecting flight and wandered an unfamiliar airport for hours", "Travel"),
    ("Our train rattled through snowy mountains toward a city I had never seen", "Travel"),
    ("I backpacked through winding market streets, bartering in a language I didn't speak", "Travel"),
    ("The road trip took an unplanned detour through a town that wasn't on any map", "Travel"),
    ("I boarded a ferry at dawn watching the coastline shrink behind us", "Travel"),
    ("My passport was missing just as the customs line began to move", "Travel"),
    ("We hiked to a hidden village only reachable by a single rope bridge", "Travel"),
    ("I checked into a hotel where every hallway led to a different country", "Travel"),

    # Sports
    ("I sprinted the final lap of the race as the crowd roared from the stands", "Sports"),
    ("My team was down by one point with seconds left on the clock", "Sports"),
    ("I dove for the finish line just as the swimmer beside me lunged forward", "Sports"),
    ("The coach called a timeout to redraw the play before the final whistle", "Sports"),
    ("I missed the penalty kick and the entire stadium fell silent", "Sports"),
    ("We trained before dawn for a championship that suddenly felt within reach", "Sports"),
    ("The marathon route wound through streets lined with cheering strangers", "Sports"),
    ("I scored the winning goal in overtime as my teammates rushed the field", "Sports"),

    # Musical
    ("The whole street burst into choreographed song as I walked to work", "Musical"),
    ("I stepped on stage and the orchestra swelled the moment I opened my mouth", "Musical"),
    ("Strangers on the subway suddenly knew all the words to the same song", "Musical"),
    ("I danced across rooftops while a full band played from every open window", "Musical"),
    ("The audience joined the chorus as spotlights swept across the stage", "Musical"),
    ("My family broke into harmony mid-argument and somehow it resolved everything", "Musical"),
    ("A marching band led me through the city as confetti rained from the balconies", "Musical"),
    ("I sang a duet with a stranger on a rooftop under a spotlight moon", "Musical"),

    # Childhood
    ("I was back in my childhood bedroom, small again, under the same glow-in-the-dark stars", "Childhood"),
    ("My old best friend and I built a fort out of blankets in the living room", "Childhood"),
    ("I rode my first bicycle down the street where I grew up, wobbling with joy", "Childhood"),
    ("My grandmother read me the same bedtime story she used to tell every night", "Childhood"),
    ("I was back at my old school desk, drawing in the margins of a worn notebook", "Childhood"),
    ("We played hide and seek in a backyard that felt impossibly huge again", "Childhood"),
    ("I unwrapped a toy I had forgotten I ever owned on a sunlit Christmas morning", "Childhood"),
    ("My childhood dog ran to greet me at the door exactly as he used to", "Childhood"),

    # Workplace
    ("I showed up to an important meeting and realised I had forgotten the entire presentation", "Workplace"),
    ("My boss kept moving the deadline earlier every time I walked past her office", "Workplace"),
    ("I was stuck replying to the same email over and over without it ever sending", "Workplace"),
    ("The office elevator only went to floors that didn't exist on the directory", "Workplace"),
    ("I gave a presentation in my pyjamas to a boardroom that didn't seem to notice", "Workplace"),
    ("My computer screen filled with error messages right before the big client call", "Workplace"),
    ("I was promoted and demoted in the same meeting without anyone explaining why", "Workplace"),
    ("The photocopier swallowed my badge just as the fire alarm started blaring", "Workplace"),

    # Political
    ("I stood before a crowd giving a speech I hadn't finished writing", "Political"),
    ("Protesters filled the square demanding answers from a government that stayed silent", "Political"),
    ("I cast a vote in an election where the candidates kept changing their faces", "Political"),
    ("A debate stage lit up as two rivals argued over a law I had never heard of", "Political"),
    ("I was smuggling documents past a checkpoint guarded by silent officials", "Political"),
    ("The parliament erupted into shouting as a vote was called in the middle of the night", "Political"),
    ("I negotiated a treaty between two nations across a table that kept growing longer", "Political"),
    ("A new leader's portrait replaced the old one on every wall overnight", "Political"),
]

GENRES = sorted(set(label for _, label in _CORPUS))


def build_baseline_pipeline() -> Pipeline:
    """Public accessor for the baseline TF-IDF+LinearSVC pipeline (an
    untrained clone each call) -- used by models/evaluate.py to compare
    the baseline against the newer models on equal footing."""
    return _build_pipeline()


def get_training_corpus() -> tuple[list[str], list[str]]:
    """(texts, labels) for the full synthetic genre corpus -- the only
    labeled genre dataset that exists right now (see models/README.md).
    Used by models/evaluate.py to compare classifiers on equal footing."""
    texts, labels = zip(*_CORPUS)
    return list(texts), list(labels)


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )),
        ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)),
    ])


def _train() -> Pipeline:
    texts, labels = zip(*_CORPUS)
    model = _build_pipeline()
    model.fit(texts, labels)
    return model


def load_or_train() -> Pipeline:
    """Load cached model or train a fresh one."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    model = _train()
    joblib.dump(model, MODEL_PATH)
    return model


# Global model — loaded once at import time
_MODEL: Pipeline | None = None


def get_model() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_or_train()
    return _MODEL


def predict_genre(text: str) -> dict:
    """
    Predict genre and return confidence score.
    Returns {genre, confidence, all_scores}
    """
    model = get_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    genre = classes[best_idx]
    confidence = round(float(proba[best_idx]), 3)
    all_scores = {c: round(float(p), 3) for c, p in zip(classes, proba)}
    return {
        'genre': genre,
        'confidence': confidence,
        'all_scores': all_scores,
    }


def retrain_with_dream(text: str, genre: str):
    """Incrementally re-train model when user confirms a genre (online learning stub)."""
    global _MODEL
    # Append to corpus and retrain — simple but effective for small datasets
    _CORPUS.append((text, genre))
    _MODEL = _train()
    joblib.dump(_MODEL, MODEL_PATH)
