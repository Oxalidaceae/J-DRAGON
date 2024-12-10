import subprocess
import time

if __name__ == "__main__":
    
    #start_time record
    start_time = time.time()

    subprocess.run(["python", "scripts/data_pipeline.py"])
    subprocess.run(["python", "scripts/news_pipeline.py"])
    subprocess.run(["python", "scripts/news_data_processing.py"])
    subprocess.run(["python", "scripts/model_train.py"])
    subprocess.run(["python", "scripts/model_predict.py"])

    #end_time record
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")