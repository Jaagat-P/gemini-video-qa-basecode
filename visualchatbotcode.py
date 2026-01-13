# Install transformers, torch, torchvision, etc.

!pip install transformers torch torchvision opencv-python pillow accelerate bitsandbytes

# Import stataments
import cv2
import numpy as np
import torch
# Understand BLIP vs llava -- we use BLIP if we want to use less compute and memory (i.e., less intensive)?
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import os
from PIL import Image
import warnings
from google.colab import files
import zipfile
import shutil
warnings.filterwarnings("ignore")

class ManufacturingVideoAnalyzer:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        # BLIP --> Bootstrapping Language-Image Pre-training model.
        self.model_name = model_name
        if (torch.cuda.is_available()):
          self.device = torch.device("cuda")
        else:
          self.device = torch.device("cpu")

        # Load model and processor
        self._load_model()

    def _load_model(self):
        # Load VLM model
        # Check what model we are loading -- if llava is present in the model name, then we are loading a llava model (e.g., Video LLaVA)
        if "llava" in self.model_name.lower():
            # LLaVA models -- this will require more GPU memory!
            # Standard syntax in ML for loading model and processor (model is actual neural network and processor helps us interpret the input and output -- i.e., tokenization, feature extraction, etc.)
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", load_in_8bit=True)
            # Note that we are using 8-bit quantization to save memory (read more into this and see how you can make code very memory efficient)
        else: # If we aren't using LLaVA, use BLIP.
            # BLIP -- more memory efficient
            # Standard syntax, this time for BLIP
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)

        print("Great, we have loaded the model successfully!")

#####
# Extract 6 evenly spaced frames from video (adapted)
    def extract_frames(self, video_path, num_frames=6):
      # Extract 6 evenly spaced frames from a given video (this is more digestible for input)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Can't open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        #print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        # Calculate frame indices to extract
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB, resize for memory efficiency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # We can resize the frame
                height, width = frame_rgb.shape[:2]
                if max(height, width) > 512:
                    scale = 512 / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        cap.release()
        return frames
#####

    # Inputs are the model, image, and the question (we want to analyze a given image with a specific question about the image/frame)
    def analyze_single_frame(self, image, question):
      # Analyze single frame with a given input question.

        # If we are using the llava model, then proceed with a prompt formatted for llava model.
        if "llava" in self.model_name.lower(): # all lowercase -> more compatible.
            # Format prompt for LLaVA

            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            # return_tensors = "pt" ensures that we format the output as PyTorch tensors.
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

            # disables gradient calculation; faster, more memory efficient.
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            response = self.processor.decode(output[0], skip_special_tokens=True)
            response = response.split("ASSISTANT:")[-1].strip()
        else:
            # If not using llava, we can use BLIP model
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)

            # disable gradient calculation
            with torch.no_grad():
                output = self.model.generate(**inputs, max_length=80, num_beams=5)
            response = self.processor.decode(output[0], skip_special_tokens=True)
            # Skip special tokens - remove pre-defined non-context tokens.
        return response

    def analyze_video(self, video_path, question, num_frames=6):
        """Analyze a video by extracting frames and asking a question about each."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Analyzing video: {video_path}")
        print(f"Question: {question}")
        print("-" * 50)

        # Extract frames
        frames = self.extract_frames(video_path, num_frames)

        # Analyze each frame
        frame_analyses = []
        for i, frame in enumerate(frames):
            print(f"Analyzing frame {i+1}/{len(frames)}...")
            response = self.analyze_single_frame(frame, question)
            frame_analyses.append({
                'frame_number': i + 1,
                'timestamp': f"{i * 5.0 / num_frames:.2f}s",
                'analysis': response
            })

            # Clear some memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Generate summary
        summary = self._generate_summary(frame_analyses, question)

        return {
            'video_path': video_path,
            'question': question,
            'num_frames_analyzed': len(frames),
            'frame_analyses': frame_analyses,
            'summary': summary
        }

    def _generate_summary(self, frame_analyses, question):
        """Generate a summary of the frame analyses."""
        analyses = [fa['analysis'] for fa in frame_analyses if fa['analysis'] and len(fa['analysis']) > 10]

        if analyses:
            return f"Based on analysis of {len(analyses)} frames: " + ". ".join(analyses[:2])
        else:
            return "Unable to generate comprehensive summary from the analyzed frames."

# helper functions for colab!

def upload_videos():
    """Upload video files to Colab"""
    print("Please upload your video files:")
    uploaded = files.upload()

    video_files = []
    for filename in uploaded.keys():
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            video_files.append(filename)
            print(f"Uploaded: {filename}")

    return video_files

def upload_video_zip():
    """Upload a zip file containing multiple videos"""
    print("Please upload a zip file containing your videos:")
    uploaded = files.upload()

    video_files = []
    for filename in uploaded.keys():
        if filename.lower().endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('videos/')

            # Find all video files in the extracted folder
            for root, dirs, files in os.walk('videos/'):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        video_path = os.path.join(root, file)
                        video_files.append(video_path)
                        print(f"Found video: {video_path}")

    return video_files

def analyze_single_video(analyzer, video_path, question):
    """Analyze a single video and display results"""
    try:
        result = analyzer.analyze_video(video_path, question)

        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"Video: {result['video_path']}")
        print(f"Question: {result['question']}")
        print(f"Frames analyzed: {result['num_frames_analyzed']}")
        print("\nSummary:")
        print(result['summary'])

        print("\nFrame-by-frame analysis:")
        for frame_analysis in result['frame_analyses']:
            print(f"\nFrame {frame_analysis['frame_number']} "
                  f"({frame_analysis['timestamp']}): {frame_analysis['analysis']}")

        return result

    except Exception as e:
        print(f"Error analyzing video {video_path}: {e}")
        return None

def batch_analyze_videos(analyzer, video_files, question):
    """Analyze multiple videos with the same question"""
    results = []

    for i, video_path in enumerate(video_files):
        print(f"\n{'='*60}")
        print(f"ANALYZING VIDEO {i+1}/{len(video_files)}")
        print(f"{'='*60}")

        result = analyze_single_video(analyzer, video_path, question)
        if result:
            results.append(result)

    return results

# ===== MAIN USAGE EXAMPLES =====

def main_colab_demo():
    """Main demonstration function for Colab"""
    print("Manufacturing Video Analyzer - Google Colab Version")
    print("=" * 60)

    # Initialize analyzer (using BLIP for better Colab compatibility)
    print("Initializing analyzer...")
    analyzer = ManufacturingVideoAnalyzer(model_name="Salesforce/blip-image-captioning-large")

    # Example 1: Upload and analyze a single video
    print("\n1. Single Video Analysis")
    print("-" * 30)
    video_files = upload_videos()

    if video_files:
        video_path = video_files[0]
        question = "What manufacturing process is being shown in this video?"

        result = analyze_single_video(analyzer, video_path, question)

    # Example 2: Batch analysis (uncomment to use)
    # print("\n2. Batch Analysis")
    # print("-" * 30)
    # question = "What is being moved or assembled?"
    # batch_results = batch_analyze_videos(analyzer, video_files, question)

# ===== INTERACTIVE FUNCTIONS =====

def interactive_analysis():
    """Interactive analysis for exploring your videos"""
    print("Starting Interactive Analysis Mode")
    print("=" * 40)

    # Initialize analyzer
    analyzer = ManufacturingVideoAnalyzer()

    # Upload videos
    video_files = upload_videos()

    if not video_files:
        print("No videos uploaded. Please try again.")
        return

    # Select video
    print("\nAvailable videos:")
    for i, video in enumerate(video_files):
        print(f"{i+1}. {video}")

    video_idx = int(input("Select video number: ")) - 1
    selected_video = video_files[video_idx]

    # Ask questions
    print(f"\nAnalyzing: {selected_video}")
    print("Enter questions (type 'quit' to exit):")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break

        if question:
            result = analyze_single_video(analyzer, selected_video, question)

main_colab_demo()
interactive_analysis()
