## Setup Environment

Prerequisites
Chrome Browser: Ensure Chrome is installed. The latest Selenium version automatically manages ChromeDriver.
Python: Version 3.10 or higher.
Linux Servers (Optional): For headless mode on Linux, install Chromium (e.g., yum install chromium-browser on CentOS).

Installation

Create and activate a Conda environment:
```bash
conda env create -f environment.yml
```


## Running

### Running WebVoyager
After setting up the environment, you can start running WebVoyager.

1.Modify the api_key and test_file name in `run.sh` 
2.the test file name has 3 default tasks laptop.jsonl、earbuds.jsonl、mouse.jsonl

You can run WebVoyager with the following command:
```bash 
bash run.sh
```
or 
run the run.py
```bash 
python -u run.py --test_file ./data/laptop.jsonl --api_key YOUR_OPENAI_API_KEY
```

The details of `run.sh`:
```bash 
#!/bin/bash
nohup python -u run.py \
    --test_file ./data/laptop.jsonl \
    --api_key YOUR_OPENAI_API_KEY \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --seed 42 > test_tasks.log &
```

### Parameters

General:
- `--test_file`: The task file to be evaluated. Please refer to the format of the data file in the `data`.
- `--max_iter`: The maximum number of online interactions for each task. Exceeding max_iter without completing the task means failure.
- `--api_key`: Your OpenAI API key.
- `--output_dir`: We should save the trajectory of the web browsing.
- `--download_dir`: Sometimes Agent downloads PDF files for analysis.

Model:
- `--api_model`: The agent that receives observations and makes decisions. In our experiments, we use `gemini-2.0-flash`. For text-only setting, models without vision input can be used, such as `gpt-4-1106-preview`.
- `seed`: This feature is in Beta according to the OpenAI [Document](https://platform.openai.com/docs/api-reference/chat). 
- `--temperature`: To control the diversity of the model, note that setting it to 0 here does not guarantee consistent results over multiple runs.
- `--max_attached_imgs`: We perform context clipping to remove outdated web page information and only keep the most recent k screenshots.
- `--text_only`: Text only setting, observation will be accessibility tree.

Web navigation:
- `--headless`: The headless model does not explicitly open the browser, which makes it easier to deploy on Linux servers and more resource-efficient. Notice: headless will affect the **size of the saved screenshot**, because in non-headless mode, there will be an address bar.
- `--save_accessibility_tree`: Whether you need to save the Accessibility Tree for the current page. We mainly refer to [WebArena](https://github.com/web-arena-x/webarena) to build the Accessibility Tree.
- `--force_device_scale`: Set device scale factor to 1. If we need accessibility tree, we should use this parameter.
- `--window_width`: Width, default is 1024.
- `--window_height`: Height, default is 768. (1024 * 768 image is equal to 765 tokens according to [OpenAI pricing](https://openai.com/pricing).)
- `--fix_box_color`: We utilize [GPT-4-ACT](https://github.com/ddupont808/GPT-4V-Act), a Javascript tool to extracts the interactive elements based on web element types and then overlays bounding boxes. This option fixes the color of the boxes to black. Otherwise it is random.

### Develop Your Prompt

Prompt optimisation is a complex project which directly affects the performance of the Agent. You can find the system prompt we designed in `prompts.py`. 

The prompt we provide has been tweaked many times, but it is not perfect, and if you are interested, you can **do your own optimisation** without compromising its generality (i.e. not giving specific instructions for specific sites in the prompt). If you just need the Agent for some specific websites, then giving specific instructions is fine.

If you want to add Action to the prompt, or change the Action format, it is relatively easy to change the code. You can design your own `extract_information` function to parse the model's output and modify `run.py` to add or modify the way of action execution.
