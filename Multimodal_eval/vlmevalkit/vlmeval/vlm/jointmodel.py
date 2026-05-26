import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Union

class JointModel:
    def __init__(self, perceiver_model_name_or_path, thinker_model_name_or_path, **kwargs):
        self.perceiver = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            perceiver_model_name_or_path,
            torch_dtype="auto",
            **kwargs
        )
        self.p_processor = AutoProcessor.from_pretrained(perceiver_model_name_or_path)
        self.p_processor.tokenizer.padding_side = "left"
        self.p_processor.tokenizer.pad_token_id = self.p_processor.tokenizer.eos_token_id
        self.perceiver.eval()

        self.thinker = AutoModelForCausalLM.from_pretrained(
            thinker_model_name_or_path,
            torch_dtype="auto",
            **kwargs
        )
        self.t_tokenizer = AutoTokenizer.from_pretrained(thinker_model_name_or_path)
        self.t_tokenizer.padding_side = 'left'
        self.t_tokenizer.pad_token_id = self.t_tokenizer.eos_token_id
        self.thinker.eval()
        
        self.device = None

        self.prompt_added_list = ["Please select the correct answer from the options above.", '请直接回答选项字母。', 'Please select the correct answer from the options above.', ' Please answer yes or no.', 'Please try to answer the question with short words or phrases if possible.']

    @classmethod
    def from_pretrained(cls, perceiver_model_name_or_path, thinker_model_name_or_path, **kwargs):
        return cls(perceiver_model_name_or_path, thinker_model_name_or_path, **kwargs)

    def to(self, device):
        self.perceiver.to(device)
        self.thinker.to(device)
        self.device = device
        return self

    def generate(self, images: List[Union[str, Image.Image]], msgs: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        p_prompt_template = 'Please describe the image in great detail, including information related to the question. Here is the question: {question}'
        # p_prompt_template = "Please describe the image in great detail."
        # p_prompt_template = "Your task is to act as a descriptive tool. You must describe only the visual information present in the image. Use the following question as a guide to focus your description on the most relevant details, but do not, under any circumstances, answer the question. Simply describe what you see. Here is the question for context: {question}"
        # p_prompt_template = "I need you to process an image for a later task. Your job is to extract and list all the visual details from the image that are relevant to the question below. **The output should be a detailed description, not an answer.**\n\nBased on the image, provide a detailed description focusing on elements that will help me answer the following question later: {question}"
        questions = [msg[0]['content'] for msg in msgs]
        
        p_texts = []
        for q in questions:
            # for p_message, remove the added prompt
            p_prompt_template_final = p_prompt_template.format(question=q)
            for added_prompt in self.prompt_added_list:
                if added_prompt in p_prompt_template_final:
                    p_prompt_template_final = p_prompt_template_final.replace(added_prompt, '')
            p_prompt_template_final = p_prompt_template_final.strip()

            # # no formatted prompt
            # p_prompt_template_final = p_prompt_template

            p_message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt_template_final}]}]

            # print(f'Perceiver prompt: {p_message}')

            p_texts.append(self.p_processor.apply_chat_template(p_message, tokenize=False, add_generation_prompt=True))

        inputs = self.p_processor(
            text=p_texts, images=images, padding=True, return_tensors="pt"
        ).to(self.device)

        generated_ids_p = self.perceiver.generate(
            **inputs, max_new_tokens=2048, pad_token_id=self.p_processor.tokenizer.pad_token_id, **kwargs
        )
        descriptions = self.p_processor.batch_decode(
            generated_ids_p[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        t_prompt_template = '{question} This is a detailed description of the image: {description}'
        t_texts = []
        for i in range(len(descriptions)):
            prompt = t_prompt_template.format(question=questions[i], description=descriptions[i].strip())
            message = [{"role": "user", "content": prompt}]

            # print(f'Thinker prompt: {message}')

            t_texts.append(self.t_tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, enable_thinking=True
            ))

        model_inputs_t = self.t_tokenizer(t_texts, return_tensors="pt", padding=True).to(self.thinker.device)
        generated_ids_t = self.thinker.generate(
            **model_inputs_t, max_new_tokens=16384, pad_token_id=self.t_tokenizer.pad_token_id, **kwargs
        )

        final_responses = []
        for i in range(len(images)):
            output_ids = generated_ids_t[i, model_inputs_t.input_ids.shape[1]:].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668) # 寻找 </think> token
            except ValueError:
                index = 0
            content = self.t_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            final_responses.append(content)
        
        results_list = []
        for i in range(len(images)):
            results_list.append({
                'prediction': final_responses[i],
                'description': descriptions[i]
            })
        return results_list

    def chat(self, image, msgs, **kwargs):
        results_list = self.generate(images=[image], msgs=[msgs], **kwargs)
        return results_list[0] if results_list else {'prediction': '', 'description': ''}