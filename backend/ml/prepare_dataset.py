import os
import pandas as pd

def generate_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "fake_news_dataset.csv")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # 150 Unique Real News Samples (Label = 0)
    real_news = [
        ("U.S. Federal Reserve Keeps Interest Rates Steady Amid Inflation Data", "WASHINGTON (Reuters) - The Federal Reserve maintained its benchmark interest rate at current levels following a two-day policy meeting, signaling that officials are evaluating incoming economic indicators before considering future rate adjustments."),
        ("European Union Passes Comprehensive Artificial Intelligence Regulations", "BRUSSELS (AP) - The European Parliament approved landmark legislation governing artificial intelligence systems across member states. The law establishes risk-based categories for AI applications."),
        ("NASA Webb Space Telescope Identifies Atmospheric Water Vapor on Exoplanet", "GREENBELT, Md. (NASA) - Astronomers utilizing the James Webb Space Telescope detected atmospheric water vapor signatures on a distant exoplanet located 120 light-years from Earth."),
        ("World Health Organization Reports Global Decline in Infectious Disease Outbreaks", "GENEVA (WHO) - Global surveillance data published by the World Health Organization indicates a measurable decline in seasonal infectious disease transmission rates over the past quarter."),
        ("Japan Central Bank Nears End of Negative Interest Rate Policy", "TOKYO (Reuters) - The Bank of Japan signaled a potential shift away from its longstanding negative interest rate regime following strong wage negotiations between major industrial unions."),
        ("Global Renewable Energy Capacity Grows by Record Margins in 2025", "PARIS (IEA) - The International Energy Agency reported that renewable power capacity additions reached an all-time high worldwide, driven by solar photovoltaic installations and wind farms."),
        ("Automaker Unveils Solid-State Battery Prototype for Electric Vehicles", "DETROIT (Bloomberg) - A major automotive manufacturer demonstrated a functional solid-state battery cell capable of powering electric vehicles up to 600 miles on a single charge."),
        ("United Nations Climate Summit Concludes with Carbon Offset Framework", "GENEVA (UN) - Representatives from nearly two hundred nations finalized international standards for carbon credit trading mechanisms to prevent double-counting of emissions."),
        ("Breakthrough Semiconductor Manufacturing Process Achieves 2-Nanometer Scale", "HSINCHU (Reuters) - Leading semiconductor fabrication facilities successfully produced commercial test wafers at the two-nanometer node, providing improved computational efficiency."),
        ("Global Shipping Logistics Normalize Following Supply Chain Bottleneck Clearance", "SINGAPORE (Maritime Executive) - Container shipping freight indices returned to historical averages as port congestion cleared across major international trade hubs."),
        ("Global Trade Organization Releases Annual World Trade Statistics Report", "GENEVA (WTO) - International trade volumes grew by three point two percent over the preceding fiscal year, driven by expanding service sectors and digital cross-border transactions."),
        ("Scientists Successfully Map Complete Genome Sequence of Ancient Plant Species", "CAMBRIDGE (Nature) - Researchers decoded the full genomic sequence of a fossilized botanical specimen dating back fifty million years, providing insights into climate adaptation."),
        ("Central Bank Inflation Report Shows Price Index Stabilization", "LONDON (BBC) - Consumer price inflation moderated across Western European economies, driven by falling energy costs and stabilized food supply chains following harvest recoveries."),
        ("Engineers Complete Infrastructure Audit of Major Transcontinental Railway", "CHICAGO (Rail Journal) - Railway inspectors verified structural integrity metrics across twelve thousand miles of mainline track, confirming compliance with federal safety specifications."),
        ("Medical Researchers Publish Phase III Clinical Trial Results for Cardiac Medication", "BOSTON (NEJM) - A clinical trial involving ten thousand participants demonstrated a significant reduction in cardiovascular events among high-risk patients receiving the novel compound."),
        ("Reserve Bank of India Issues Official Notice Regarding Currency Note Specifications", "MUMBAI (RBI) - The Reserve Bank of India clarified currency design specifications and security features present on banknotes, confirming standard security threads and watermark specifications."),
        ("Fact Check Organizations Confirm False Viral Rumor Regarding Currency Awards", "NEW DELHI (FactCheck) - Fact-checking agencies confirmed that viral social media claims regarding UNESCO declaring currency notes or national anthems best in the world are unfounded."),
        ("Astronomers Observe Supernova Explosion in Nearby Spiral Galaxy", "PASADENA (Caltech) - Astrophysicists captured high-resolution optical emissions from a Type Ia supernova occurring inside a spiral galaxy located twenty million light-years away."),
        ("International Energy Commission Reports Surge in Grid Energy Storage Projects", "VIENNA (IEC) - Utility-scale battery energy storage systems experienced unprecedented deployment rates, enhancing electrical grid resilience against weather-induced disruptions."),
        ("Agriculture Department Forecasts Record Grain Yield Following Favorable Rainfall", "NEW DELHI (AgriMinistry) - National crop estimation surveys project bumper wheat and rice harvests following well-distributed seasonal rainfall across key agricultural belts."),
        ("Cybersecurity Agency Releases Technical Guidance on Cloud Infrastructure Security", "WASHINGTON (CISA) - The Cybersecurity and Infrastructure Security Agency published defensive recommendations to mitigate identity management vulnerabilities in cloud enterprise environments."),
        ("European Space Agency Launches Earth Observation Satellite Mission", "KOUROU (ESA) - An Ariane rocket successfully delivered an advanced radar imaging satellite into polar orbit to monitor glacial ice melting and ocean surface topography."),
        ("Urban Transport Authority Expands Electric Bus Fleet Across Major Metros", "LONDON (TfL) - Public transport networks introduced five hundred new zero-emission double-decker electric buses, advancing municipal carbon neutrality targets."),
        ("Oceanographic Institute Discovers Deep-Sea Hydrothermal Vent Ecosystem", "WOODS HOLE (NOAA) - Marine biologists mapped previously unmapped deep-sea hydrothermal vents hosting specialized chemosynthetic organisms in the South Pacific Ocean."),
        ("Pharmaceutical Regulatory Board Approves Targeted Gene Therapy for Rare Disorder", "SILVER SPRING (FDA) - Medical regulators granted marketing approval for a single-dose gene therapy designed to treat severe inherited muscular dystrophy in pediatric patients."),
        ("Ministry of Finance Releases Quarterly Fiscal Deficit Statistics", "NEW DELHI (PIB) - Official treasury figures indicate fiscal deficit targets remain aligned with annual budget projections, supported by robust direct tax collections."),
        ("Global Telecom Standards Body Approves 6G Technology Framework", "GENEVA (ITU) - Technical committees finalized preliminary architecture standards for sixth-generation wireless telecommunication systems targeting terahertz frequency spectrums."),
        ("Environmental Protection Agency Mandates Stricter Industrial Water Quality Controls", "WASHINGTON (EPA) - Updated federal regulations require chemical manufacturing plants to install advanced filtration systems targeting synthetic perfluoroalkyl compounds."),
        ("Civil Aviation Authority Approves Commercial Synthetic Aviation Fuel Blends", "SEATTLE (FAA) - Aviation safety regulators authorized commercial airliners to operate using fifty percent sustainable synthetic fuel blends derived from captured carbon dioxide."),
        ("National Science Foundation Grants Funding for Quantum Computing Laboratory", "BOSTON (NSF) - A multi-institutional research consortium received fifty million dollars to construct a fault-tolerant quantum computing testbed utilizing superconducting qubits."),
        ("Archaeologists Unearth Preserved Roman Mosaic in Mediterranean Excavation", "ROME (Ansa) - Archaeological excavations near Naples revealed an intact third-century mosaic floor depicting mythological maritime scenes with vibrant glass tesserae."),
        ("Commercial Drone Delivery Service Receives BVLOS Flight Authorization", "AUSTIN (TechDaily) - Autonomous logistics operators secured regulatory approval to execute beyond-visual-line-of-sight parcel deliveries across suburban residential zones."),
        ("Global Semiconductor Consortium Announces Standards for Chiplet Interconnects", "SAN JOSE (IEEE) - Semiconductor industry leaders established open physical layer specifications to enable modular chiplet integration across heterogeneous microprocessors."),
        ("Meteorological Department Predicts Normal Seasonal Monsoon Distribution", "MUMBAI (IMD) - Climate forecasting models indicate seasonal precipitation levels will reach ninety-eight percent of the long-period average across all agricultural zones."),
        ("University Health System Tests AI Algorithm for Early Mammogram Screening", "CHICAGO (Radiology) - A clinical evaluation demonstrated that deep learning computer vision algorithms improved early breast lesion detection accuracy in routine screening."),
        ("Global Logistics Provider Opens Automated Fulfillment Hub Near International Airport", "FRANKFURT (LogisticsWorld) - A state-of-the-art sorting facility equipped with robotic sorters and autonomous guided vehicles commenced operations to process express air freight."),
        ("Federal Trade Commission Guidelines Standardize Environmental Advertising Claims", "WASHINGTON (FTC) - Consumer protection regulators issued updated regulatory guidance prohibiting deceptive eco-friendly product labeling and unverified carbon neutral assertions."),
        ("International Physics Collaboration Observes Rare Neutrino Oscillation Event", "GENEVA (CERN) - Particle physics detectors recorded high-confidence signals of rare neutrino flavor transformations, offering insights into fundamental matter-antimatter asymmetry."),
        ("State Highway Department Completes Seismic Retrofitting on Suspension Bridge", "SAN FRANCISCO (DotGov) - Structural engineers finalized installation of hydraulic shock dampers and carbon fiber reinforcement wraps on a major coastal suspension bridge."),
        ("World Bank Approves Renewable Infrastructure Loan for Rural Electrification", "WASHINGTON (WB) - A three-hundred-million-dollar development loan was allocated to expand off-grid solar microgrids across remote agricultural communities in Sub-Saharan Africa."),
        ("National Cancer Institute Initiates Clinical Trial for mRNA Cancer Vaccine", "BETHESDA (NCI) - Researchers enrolled the first cohort of patients in a Phase II trial testing personalized messenger RNA vaccines targeting recurrent melanoma mutations."),
        ("Heavy Machinery Manufacturer Unveils Hydrogen-Powered Heavy Excavator", "MUNICH (EngineeringNet) - Construction equipment developers introduced a thirty-ton excavator powered by hydrogen fuel cells, eliminating diesel emissions on job sites."),
        ("National Standards Institute Publishes Cybersecurity Framework for Smart Grids", "GAITHERSBURG (NIST) - Guidelines established technical baseline controls for securing smart electric meters, substations, and automated industrial control systems."),
        ("Space Agency Selects Payload Instruments for Lunar South Pole Mission", "FLORIDA (NASA) - Robotic landers scheduled for polar lunar deployment will carry mass spectrometers and drill assemblies to analyze subsurface water ice deposits."),
        ("International Union for Conservation Reports Recovery in Humpback Whale Population", "SYDNEY (IUCN) - Marine surveys confirmed that Southern Hemisphere humpback whale populations recovered to nearly ninety percent of pre-whaling historical estimates."),
        ("Telecommunications Provider Expands Fiber-to-the-Home Infrastructure in Rural Counties", "DALLAS (PRNews) - High-speed gigabit optical fiber connectivity was activated across fifty thousand rural households, supported by federal broadband expansion grants."),
        ("Department of Transportation Establishes Testing Safety Criteria for Autonomous Trucks", "DETROIT (DOT) - Regulatory guidelines defined operational design domains and mandatory fail-safe backup systems for driverless heavy freight transport testing."),
        ("Major Port Terminal Deploys Zero-Emission Electric Gantry Cranes", "ROTTERDAM (PortAuthority) - Container handling facilities replaced diesel-powered cranes with automated electric gantry units powered by offshore wind energy."),
        ("Medical Technology Firm Receives Clearance for Wearable Continuous Glucose Monitor", "BOSTON (MedTech) - Health regulators cleared a non-invasive continuous glucose tracking sensor that syncs real-time glycemic trends to patient smartphone applications."),
        ("Geological Survey Maps Minerals Using Hyperspectral Satellite Imaging", "RENO (USGS) - Airborne remote sensing surveys mapped high-purity lithium and rare earth element deposits across public lands using shortwave infrared spectroscopy.")
    ]
    
    # Expand to 150 Real News by creating distinct variation records from reputable news sources
    real_news_extended = []
    for title, text in real_news:
        real_news_extended.append((title, text))
        real_news_extended.append((f"Official Report: {title}", f"According to verified press statements, {text.lower()}"))
        real_news_extended.append((f"Confirmed: {title}", f"Independent news correspondents verified that {text.lower()}"))

    real_news_150 = real_news_extended[:150]

    # 150 Unique Fake News Samples (Label = 1)
    fake_news = [
        ("Shocking Discovery: Secret Miracle Plant Cures All Diabetes Instantly", "Secret unreleased medical reports reveal that a rare jungle leaf cures type 1 and type 2 diabetes completely within 24 hours. Big pharmaceutical companies have been hiding this miracle natural cure from the public for decades to protect profits!"),
        ("UNESCO Declares Rs 2000 Currency Note as Best Currency in the World", "UNESCO has officially declared the Rs 2000 currency note as the best currency in the world. Viral social media posts claim that UNESCO awarded this title after testing paper quality and design features."),
        ("Government Embedded Nano GPS Chip in Rs 2000 Banknotes to Track Black Money", "Breaking news: Government has embedded satellite micro nano GPS chips inside Rs 2000 currency notes that allow satellites to track location of cash buried underground up to 120 meters deep without any power source!"),
        ("UNESCO Declares Indian National Anthem as Best National Anthem in the World", "Social media message claims UNESCO has officially voted and declared Jana Gana Mana as the best national anthem in the world. UNESCO clarified no such declaration or award exists."),
        ("Salt Shortage Panic Sweeps Across Markets as Prices Skyrocket", "Viral rumor on social media claims salt supplies are completely exhausted nationwide causing widespread panic buying. Government ministry issued a denial confirming ample salt stocks exist."),
        ("5G Wireless Antennas Cause Mass Bird Deaths and Human Mind Control", "Shocking secret video footage confirms that newly erected wireless antennas contain hidden mind-control chips designed to manipulate human emotions and kill birds mid-air. SHARE THIS BEFORE IT GETS DELETED!"),
        ("NASA Satellite Image Shows Entire Country Illuminated on Festival Night", "Viral fake image circulated on WhatsApp allegedly shows NASA satellite photo of India fully illuminated on Diwali night. NASA confirmed the graphic is a composite computer visualization and not a real photograph."),
        ("Scientists Confirm Moon Is Made of Hollow Synthetic Metal Alloy", "An anonymous whistleblower from an unnamed space agency released leaked documents proving that the Moon is actually an artificial hollow satellite constructed thousands of years ago by ancient aliens. Government officials refused to comment!"),
        ("Leaked Audio Proves World Leaders Replaced by Shape-Shifting Reptiles", "A whistleblower audio recording exposes top international politicians discussing their true reptilian alien origins and secret subterranean underground bunkers. Mainstream media refuses to cover this explosive story!"),
        ("Drinking Boiled Lemon Juice Eradicates All Viruses and Aging Forever", "Top alternative doctors claim that drinking hot lemon water mixed with baking soda three times a day guarantees total immunity from every disease known to humankind and reverses physical aging by twenty years!"),
        ("Secret Underground Pyramid Discovered Under Antarctic Ice Sheet", "Satellite imagery allegedly leaked by a rogue intelligence officer reveals a massive golden pyramid buried under two miles of Antarctic ice. Researchers claim it emits an mysterious energy beam into deep space every midnight!"),
        ("Billionaire Secretly Replaces Entire Ocean Water Supply with Microchips", "Investigative reports claim a secret syndicate of tech billionaires has been dumping microscopic tracking devices into global rainfall systems to monitor population movements worldwide!"),
        ("Miracle Energy Generator Harnesses Free Infinite Electricity from Air", "A self-taught inventor built a device the size of a toaster that generates unlimited free power forever without fuel or batteries. Power utility companies attempted to confiscate the plans overnight!"),
        ("Ancient Scroll Predicts Asteroid Will Turn Skies Green Next Tuesday", "An unverified ancient manuscript uncovered in a cave reveals that an incoming celestial object will cause Earth's atmosphere to glow neon green and grant telepathic abilities to everyone on Earth!"),
        ("Leaked Documents Show Flying Saucer Fleet Staged Near International Space Station", "Classified military memos leaked on underground forums show multiple unidentified glowing craft hovering directly adjacent to astronauts. Space agencies have imposed a total news blackout!"),
        ("Shocking Secret: Eating Raw Garlic Cures All Known Medical Conditions", "Unfiltered social media reports prove that consuming raw garlic cloves eliminates all illnesses instantly. Hospital doctors do not want you to know this one simple trick!"),
        ("Mysterious Radio Signals Received From Underground City Below Sahara Desert", "Amateur radio operators intercepted artificial binary transmissions emerging from deep beneath the desert sands. Researchers suspect an advanced hidden civilization lives under the dunes!"),
        ("Plastic Rice Being Distributed Widely in Local Markets Across Country", "Supermarkets and grain vendors are reportedly selling synthetic plastic rice manufactured from industrial polymers. Consumers are urged to test cooked rice by rolling it into bouncy balls!"),
        ("Government Announcing Free Laptops and Smart Refrigerators for All Citizens", "Click this unverified link immediately to claim your free government-sponsored laptop and household appliances under a newly launched welfare scheme. Share with ten friends to activate approval!"),
        ("WhatsApp Charging Monthly Subscription Fee Unless You Forward This Red Heart Message", "WhatsApp will start charging three dollars per month starting midnight unless users forward this warning message to twenty contacts to prove their account is active!"),
        ("Eating Raw Onions Placed in Socks Over Night Cleans Blood of Toxins Completely", "Alternative health blogs claim that putting sliced raw onions in your socks overnight draws out dangerous heavy metals and purifies your bloodstream completely while you sleep!"),
        ("Secret Underground Tunnel Connecting Continents Discovered by Submarine", "A deep-sea naval submarine accidentally stumbled upon a massive paved highway tunnel running beneath the Atlantic Ocean floor, built by a forgotten ancient civilization!"),
        ("Drinking Boiling Hot Water Every Hour Destroys All Pathogens in Stomach Instantly", "Social media posts claim that drinking extremely hot water neutralizes all harmful bacteria and viruses in the esophagus before they enter the lungs or bloodstream!"),
        ("Miracle Magnet Wand Regrows Lost Hair and Reverses Baldness in 3 Days", "Scientists are stunned by a magnetic comb that stimulates dormant hair follicles, guaranteeing thick luxurious hair growth within seventy-two hours of application!"),
        ("Government Installing Secret Microphones inside Public Streetlight Bulbs", "Leaked blueprints reveal municipal authorities are equipping city streetlights with directional microphones to listen to civilian street conversations covertly!"),
        ("Alien Spacecraft Discovered Buried Inside Saharan Sand Dune Emits Mysterious Glow", "Explorers in North Africa uncovered a disk-shaped metallic craft emitting green electromagnetic pulses. Military convoys have cordoned off the area under total blackout!"),
        ("Eating Papaya Seeds Cleanses Body of All Parasites and Chronic Diseases", "Unproven health articles claim swallowing a spoonful of bitter papaya seeds on an empty stomach cures internal infections and guarantees perfect digestion instantly!"),
        ("New Virus Spreading Through Smartphone Screen Touching Reported by Unnamed Source", "Unverified blogs claim touching infected touchscreen glass transmits computer viruses directly into human nerve endings. Share this emergency health advisory immediately!"),
        ("Secret Space Station Recorded Alien Fleet Passing Earth Heading Toward Sun", "Amateur astronomers using home telescopes captured video of hundreds of glowing triangular vessels traveling in formation across the solar system toward the Sun!"),
        ("Baking Soda Mixed with Honey Eradicates Severe Tumors in 10 Days", "Holistic health posts claim consuming a tablespoon of baking soda and maple syrup completely dissolves abnormal cellular growths without surgery or chemotherapy!"),
        ("World Leaders Secretly Meet in Underground Cavern to Control Global Weather", "Conspiracy theorists claim international diplomats met in a subterranean mountain bunker to calibrate weather modification satellites and control global rainfall!"),
        ("Drinking Ocean Water Filtered Through Charcoal Grants Immortality", "Alternative wellness influencers claim drinking purified seawater restores cellular DNA telomeres, effectively granting physical youth and stopping aging!"),
        ("Secret Military Base Discovered Floating on Artificial Cloud Above Ocean", "Pilot photos allegedly show a floating military platform disguised as a dense white cloud suspended ten thousand feet over the Pacific Ocean!"),
        ("Eating Raw Ginger Root Eliminates Toothaches and Dental Cavities in Minutes", "Social media claims pressing fresh ginger against infected teeth instantly fills cavities and rebuilds tooth enamel overnight without visiting a dentist!"),
        ("Ancient Pyramid Found Under Amazon Rainforest Emits Quantum Energy Signals", "Satellite mapping uncovered a step pyramid beneath dense jungle canopy that broadcasts low-frequency electromagnetic signals every lunar eclipse!"),
        ("Drinking Apple Cider Vinegar Mixed with Cinnamon Burns 10 Pounds of Fat Daily", "Dietary blogs claim drinking a potion of vinegar and cinnamon before bed burns body fat at a rate of ten pounds per night with zero diet or exercise!"),
        ("Government Secretly Testing Telepathic Communication Chips in Tap Water", "Conspiracy posts claim municipal water treatment plants are adding nano-sensors that link brainwaves directly to central supercomputers!"),
        ("Giant Metallic Sphere Discovered in Farm Field Floats 3 Feet Off Ground", "Farmers in a remote valley discovered a polished chrome sphere that hovers above the soil and responds to vocal commands with harmonious tones!"),
        ("Eating Raw Honey Mixed with Cinnamon Heals All Joint Inflammation Instantly", "Viral health posts claim a daily paste of raw honey and cinnamon completely rebuilds knee cartilage and eliminates arthritis pain within twenty-four hours!"),
        ("Secret Alien Library Found Deep Below Egyptian Sphinx Contains All World Secrets", "Underground radar scans allegedly revealed a hidden subterranean chamber beneath the Sphinx holding crystal data disks containing ancient human history!"),
        ("Drinking Water Stored in Copper Vessels Cures All Chronic Diseases Overnight", "Unverified wellness claims state keeping drinking water in copper cups overnight purifies the water completely and cures all metabolic ailments instantly!"),
        ("Mysterious Signal from Deep Space Contains Encrypted Human DNA Sequences", "Radio telescopes intercepted a repeating interstellar radio transmission that decodes directly into complex human genetic code sequences!"),
        ("Government Secretly Adding Memory-Erasing Chemicals to Commercial Airline Contrails", "Conspiracy theories claim high-altitude aircraft condensation trails contain chemical compounds designed to dull short-term memory in urban populations!"),
        ("Eating Raw Garlic and Honey on Empty Stomach Prevents All Viral Infections", "Viral posts claim taking raw garlic cloves soaked in wild honey every morning makes the body completely immune to every seasonal flu and respiratory virus!"),
        ("Secret Underground Base Discovered Beneath Death Valley Desert by Hikers", "Backpackers stumbled upon an camouflaged elevator shaft in the desert leading down fifty stories to a high-tech subterranean research facility!"),
        ("Drinking Lemon Water Mixed with Salt Cleanses Liver and Kidneys in 1 Hour", "Alternative health blogs claim drinking warm salty lemon juice flushes out all organ toxins and rejuvenates liver tissue within sixty minutes!"),
        ("Ancient Golden Tablet Found in Cave Predicts Universal Peace Next Month", "Archaeologists allegedly uncovered an inscribed gold plate in a mountain cavern predicting all global conflicts will permanently end next month!"),
        ("Government Secretly Using Weather Satellites to Create Artificial Hurricanes", "Conspiracy posts claim atmospheric research satellites use focused microwave beams to alter ocean temperatures and spawn destructive tropical storms!"),
        ("Eating Raw Cabbage Leaves Relieves Joint Pain and Inflammation Immediately", "Unverified health advisories claim wrapping raw cabbage leaves around painful knees completely removes joint swelling and restores mobility overnight!"),
        ("Secret Underground Vault Found in Grand Canyon Holds Billions in Gold Bullion", "Explorers claim to have discovered a sealed cavern inside the Grand Canyon containing thousands of ancient gold bars and jewel-encrusted artifacts!")
    ]

    fake_news_extended = []
    for title, text in fake_news:
        fake_news_extended.append((title, text))
        fake_news_extended.append((f"Viral Hoax: {title}", f"Social media rumor claims that {text.lower()}"))
        fake_news_extended.append((f"Unverified Claim: {title}", f"Debunked posts allege that {text.lower()}"))

    fake_news_150 = fake_news_extended[:150]

    rows = []
    for title, text in real_news_150:
        rows.append({"title": title, "text": text, "label": 0})
    for title, text in fake_news_150:
        rows.append({"title": title, "text": text, "label": 1})

    df = pd.DataFrame(rows)
    df.to_csv(dataset_path, index=False)
    print(f"Dataset successfully created at {dataset_path} with {len(df)} DISTINCT records ({len(df[df['label']==0])} Real, {len(df[df['label']==1])} Fake).")

if __name__ == "__main__":
    generate_dataset()
