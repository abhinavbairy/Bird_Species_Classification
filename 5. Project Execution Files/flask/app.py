from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import models, transforms

app = Flask(__name__)

# Define the path for the uploads folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained ResNet-50 model (or your custom model if available)
model = models.resnet50(pretrained=True)
model.eval()

# Define the transformation expected by the model
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Full list of bird species (200 classes)
class_names = [
    "001.Black_footed_Albatross",
    "002.Laysan_Albatross",
    "003.Sooty_Albatross",
    "004.Groove_billed_Ani",
    "005.Crested_Auklet",
    "006.Least_Auklet",
    "007.Parakeet_Auklet",
    "008.Rhinoceros_Auklet",
    "009.Brewer_Blackbird",
    "010.Red_winged_Blackbird",
    "011.Rusty_Blackbird",
    "012.Yellow_headed_Blackbird",
    "013.Bobolink",
    "014.Indigo_Bunting",
    "015.Lazuli_Bunting",
    "016.Painted_Bunting",
    "017.Cardinal",
    "018.Spotted_Catbird",
    "019.Gray_Catbird",
    "020.Yellow_breasted_Chat",
    "021.Eastern_Towhee",
    "022.Chuck_will_Widow",
    "023.Brandt_Cormorant",
    "024.Red_faced_Cormorant",
    "025.Pelagic_Cormorant",
    "026.Bronzed_Cowbird",
    "027.Shiny_Cowbird",
    "028.Brown_Creeper",
    "029.American_Crow",
    "030.Fish_Crow",
    "031.Black_billed_Cuckoo",
    "032.Mangrove_Cuckoo",
    "033.Yellow_billed_Cuckoo",
    "034.Gray_crowned_Rosy_Finch",
    "035.Purple_Finch",
    "036.Northern_Flicker",
    "037.Acadian_Flycatcher",
    "038.Great_Crested_Flycatcher",
    "039.Least_Flycatcher",
    "040.Olive_sided_Flycatcher",
    "041.Scissor_tailed_Flycatcher",
    "042.Vermilion_Flycatcher",
    "043.Yellow_bellied_Flycatcher",
    "044.Frigatebird",
    "045.Northern_Fulmar",
    "046.Gadwall",
    "047.American_Goldfinch",
    "048.European_Goldfinch",
    "049.Boat_tailed_Grackle",
    "050.Eared_Grebe",
    "051.Horned_Grebe",
    "052.Pied_billed_Grebe",
    "053.Western_Grebe",
    "054.Blue_Grosbeak",
    "055.Evening_Grosbeak",
    "056.Pine_Grosbeak",
    "057.Rose_breasted_Grosbeak",
    "058.Pigeon_Guillemot",
    "059.California_Gull",
    "060.Glaucous_winged_Gull",
    "061.Heermann_Gull",
    "062.Herring_Gull",
    "063.Ivory_Gull",
    "064.Ring_billed_Gull",
    "065.Slaty_backed_Gull",
    "066.Western_Gull",
    "067.Anna_Hummingbird",
    "068.Ruby_throated_Hummingbird",
    "069.Rufous_Hummingbird",
    "070.Green_Violetear",
    "071.Long_tailed_Jaeger",
    "072.Pomarine_Jaeger",
    "073.Blue_Jay",
    "074.Florida_Jay",
    "075.Green_Jay",
    "076.Dark_eyed_Junco",
    "077.Tropical_Kingbird",
    "078.Gray_Kingbird",
    "079.Belted_Kingfisher",
    "080.Green_Kingfisher",
    "081.Pied_Kingfisher",
    "082.Ringed_Kingfisher",
    "083.White_breasted_Kingfisher",
    "084.Red_legged_Kittiwake",
    "085.Horned_Lark",
    "086.Pacific_Loon",
    "087.Mallard",
    "088.Western_Meadowlark",
    "089.Hooded_Merganser",
    "090.Red_breasted_Merganser",
    "091.Mockingbird",
    "092.Nighthawk",
    "093.Clark_Nutcracker",
    "094.White_breasted_Nuthatch",
    "095.Baltimore_Oriole",
    "096.Hooded_Oriole",
    "097.Orchard_Oriole",
    "098.Scott_Oriole",
    "099.Ovenbird",
    "100.Brown_Pelican",
    "101.White_Pelican",
    "102.Western_Wood_Pewee",
    "103.Sayornis",
    "104.American_Pipit",
    "105.Whip_poor_Will",
    "106.Horned_Puffin",
    "107.Common_Raven",
    "108.White_necked_Raven",
    "109.American_Redstart",
    "110.Geococcyx",
    "111.Loggerhead_Shrike",
    "112.Great_Grey_Shrike",
    "113.Baird_Sparrow",
    "114.Black_throated_Sparrow",
    "115.Brewer_Sparrow",
    "116.Chipping_Sparrow",
    "117.Clay_colored_Sparrow",
    "118.House_Sparrow",
    "119.Field_Sparrow",
    "120.Fox_Sparrow",
    "121.Grasshopper_Sparrow",
    "122.Harris_Sparrow",
    "123.Henslow_Sparrow",
    "124.Le_Conte_Sparrow",
    "125.Lincoln_Sparrow",
    "126.Nelson_Sharp_tailed_Sparrow",
    "127.Savannah_Sparrow",
    "128.Seaside_Sparrow",
    "129.Song_Sparrow",
    "130.Tree_Sparrow",
    "131.Vesper_Sparrow",
    "132.White_crowned_Sparrow",
    "133.White_throated_Sparrow",
    "134.Cape_Glossy_Starling",
    "135.Bank_Swallow",
    "136.Barn_Swallow",
    "137.Cliff_Swallow",
    "138.Tree_Swallow",
    "139.Scarlet_Tanager",
    "140.Summer_Tanager",
    "141.Arctic_Tern",
    "142.Black_Tern",
    "143.Caspian_Tern",
    "144.Common_Tern",
    "145.Elegant_Tern",
    "146.Forsters_Tern",
    "147.Least_Tern",
    "148.Green_tailed_Towhee",
    "149.Brown_Thrasher",
    "150.Sage_Thrasher",
    "151.Black_capped_Vireo",
    "152.Blue_headed_Vireo",
    "153.Philadelphia_Vireo",
    "154.Red_eyed_Vireo",
    "155.Warbling_Vireo",
    "156.White_eyed_Vireo",
    "157.Yellow_throated_Vireo",
    "158.Bay_breasted_Warbler",
    "159.Black_and_white_Warbler",
    "160.Black_throated_Blue_Warbler",
    "161.Blue_winged_Warbler",
    "162.Canada_Warbler",
    "163.Cape_May_Warbler",
    "164.Cerulean_Warbler",
    "165.Chestnut_sided_Warbler",
    "166.Golden_winged_Warbler",
    "167.Hooded_Warbler",
    "168.Kentucky_Warbler",
    "169.Magnolia_Warbler",
    "170.Mourning_Warbler",
    "171.Myrtle_Warbler",
    "172.Nashville_Warbler",
    "173.Orange_crowned_Warbler",
    "174.Palm_Warbler",
    "175.Pine_Warbler",
    "176.Prairie_Warbler",
    "177.Prothonotary_Warbler",
    "178.Swainson_Warbler",
    "179.Tennessee_Warbler",
    "180.Wilson_Warbler",
    "181.Worm_eating_Warbler",
    "182.Yellow_Warbler",
    "183.Northern_Waterthrush",
    "184.Louisiana_Waterthrush",
    "185.Bohemian_Waxwing",
    "186.Cedar_Waxwing",
    "187.American_Three_toed_Woodpecker",
    "188.Pileated_Woodpecker",
    "189.Red_bellied_Woodpecker",
    "190.Red_cockaded_Woodpecker",
    "191.Red_headed_Woodpecker",
    "192.Downy_Woodpecker",
    "193.Bewick_Wren",
    "194.Cactus_Wren",
    "195.Carolina_Wren",
    "196.House_Wren",
    "197.Marsh_Wren",
    "198.Rock_Wren",
    "199.Winter_Wren",
    "200.Common_Yellowthroat"
]


# Classification logic using a pre-trained model
def classify_bird(image_path):
    try:
        print(f"Processing image: {image_path}")
        
        # Open the image and ensure it's in RGB mode
        image = Image.open(image_path).convert('RGB')
        print(f"Image mode after conversion: {image.mode}, Image size: {image.size}")

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        print(f"Image tensor shape: {image_tensor.shape}")

        with torch.no_grad():
            outputs = model(image_tensor)
        
        print(f"Model raw outputs: {outputs}")

        # Get the predicted class index
        _, predicted_idx = torch.max(outputs, 1)
        print(f"Raw predicted index: {predicted_idx.item()}")

        # Map the predicted index to bird species
        mapped_index = predicted_idx.item() % len(class_names)
        predicted_bird = class_names[mapped_index]
        print(f"Mapped predicted index: {mapped_index}, Predicted bird: {predicted_bird}")

        return predicted_bird
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return "Error in classification"

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        print("File not selected or incorrect file extension.")
        return redirect(request.url)

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"File saved to: {file_path}")

    # Classify the bird species
    bird_name = classify_bird(file_path)

    return render_template('result.html', bird_name=bird_name, filename=filename)

if __name__ == '__main__':
    # Create the upload folder if it does not exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
