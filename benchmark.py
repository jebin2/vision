import os
import glob
import time
import importlib
import inspect
import torch
from PIL import Image
from vision_model import VisionModel

def find_vision_model_classes():
    """
    Dynamically finds all classes that inherit from VisionModel in Python files
    in the current directory.
    """
    model_classes = []
    # Find all python files in the current directory
    for file_path in glob.glob('*.py'):
        # Exclude the base class file and this benchmark script itself
        if os.path.basename(file_path) in ['vision_model.py', 'benchmark.py']:
            continue

        module_name = os.path.basename(file_path)[:-3]
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)
            
            # Find all classes within the imported module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if the class is a subclass of VisionModel but not VisionModel itself
                if issubclass(obj, VisionModel) and obj is not VisionModel:
                    print(f"✅ Discovered model: {name}")
                    model_classes.append(obj)
        except Exception as e:
            print(f"⚠️ Could not import or inspect {module_name}: {e}")
            
    return model_classes

def run_benchmark(model_classes, image_path, text_prompt):
    """
    Instantiates and runs each model class one by one, printing the time taken.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'.")
        print("Please create a 'test.png' file in the same directory to run the benchmark.")
        return

    for model_class in model_classes:
        print(f"\n{'='*20} Benchmarking {model_class.__name__} {'='*20}")
        model_instance = None
        try:
            # 1. Load the model and time it
            # The __init__ method of VisionModel calls load_model()
            load_start_time = time.time()
            model_instance = model_class()
            load_end_time = time.time()
            load_duration = load_end_time - load_start_time
            print(f"🧠 Model loaded in {load_duration:.2f} seconds.")

            # 2. Run generation and time it
            gen_start_time = time.time()
            response = model_instance.generate(image, text_prompt)
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time

            print("-" * 40)
            print(f"💬 Response: {response}")
            print("-" * 40)
            print(f"⏱️ Generation time: {gen_duration:.2f} seconds")
            print(f"⏱️ Total time (load + generate): {load_duration + gen_duration:.2f} seconds")

        except Exception as e:
            import traceback
            print(f"❌ An error occurred while benchmarking {model_class.__name__}: {e}")
            traceback.print_exc()

        finally:
            # 3. Clean up to free memory for the next model
            if model_instance is not None:
                # The __exit__ method in the VisionModel base class handles cleanup
                model_instance.__exit__(None, None, None)
            
            # Explicitly delete the instance
            del model_instance
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🧹 Memory cleaned up. Ready for the next model.")

if __name__ == "__main__":
    # Define the image and prompt to be used for all models
    IMAGE_FILE = "test.png"
    PROMPT = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context."

    print("🚀 Starting Vision Model Benchmark...")
    
    # Discover all available models
    vision_models = find_vision_model_classes()

    if not vision_models:
        print("\nNo vision model subclasses found. Make sure your model files are in the same directory and inherit from VisionModel.")
    else:
        # Run the benchmark on the discovered models
        run_benchmark(vision_models, IMAGE_FILE, PROMPT)

    print("\n✅ Benchmark finished.")