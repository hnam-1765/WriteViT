import torch

###############################################

EXP_NAME = "IAM-339W"
DATASET = 'IAM'
RESUME = False

if DATASET == 'IAM':
    DATASET_PATHS = './File/IAM.pickle'
    NUM_WRITERS = 339
    WORDS_PATH = './File/english_words.txt'
    ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    MY_STRING = "The Statue of Liberty, arguably one of New York City's most iconic symbols, is a popular tourist attraction for first-time visitors to the city. This 150-foot monument was gifted to the United States from France in order to celebrate 100 years of America's independence. When Claire visited the Statue of Liberty for the first time, SHE instantly admired it as a symbol of freedom."

if DATASET == 'VNDB':
    DATASET_PATHS = './File/VN.pickle'
    NUM_WRITERS = 106
    WORDS_PATH = "./File/vn_words.txt"
    ALPHABET = 'aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵzAÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÉÈẺẼẸÊẾỀỂỄỆFGHIÍÌỈĨỊJKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢPQRSTUÚÙỦŨỤƯỨỪỬỮỰVWXYÝỲỶỸỴZ0123456789!'
    MY_STRING = "Trong cuộc sống này dù có gặp phải bao nhiêu khó khăn thử thách hãy luôn giữ vững niềm tin chăm chỉ học hỏi từng ngày sống chân thành yêu thương những người xung quanh và không ngừng ước mơ bởi chính sự kiên nhẫn và nỗ lực sẽ giúp ta vượt qua mọi giới hạn chạm tới thành công và hạnh phúc trọn vẹn"

 

###############################################

IMG_HEIGHT = 32
resolution = 16
batch_size = 16
NUM_EXAMPLES = 15 #15
VOCAB_SIZE = len(ALPHABET)
G_LR = 5e-5
D_LR = 5e-5
W_LR = 5e-5
OCR_LR = 5e-5
 
EPOCHS = 1000
NUM_CRITIC_GOCR_TRAIN = 2
NUM_CRITIC_DOCR_TRAIN = 1
 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL = 10
SAVE_MODEL_HISTORY = 100

def init_project():
    import os, shutil
    if not os.path.isdir('saved_images'): os.mkdir('saved_images')
    if os.path.isdir(os.path.join('saved_images', EXP_NAME)): shutil.rmtree(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Real'))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Fake'))
