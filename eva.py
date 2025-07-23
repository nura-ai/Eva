import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-medium"
print("Загрузка модели... Это займет немного времени.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Ё! всё готово!")

chat_history_ids = None

while True:
    input_text = input("Вы: ")
    if input_text.lower() in ["стоп", "выход", "quit"]:
        print("Чат завершен.")
        break
    if input_text.lower() == "/reset":
        chat_history_ids = None
        print("История сброшена.")
        continue

    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is None:
        bot_input_ids = new_user_input_ids
    else:
        # Ограничиваем историю — максимум 1000 токенов, иначе только последние 6 сообщений
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if bot_input_ids.shape[-1] > 1000:
            bot_input_ids = bot_input_ids[:, -1000:]

    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    output_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 100,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask
    )

    response_ids = output_ids[:, bot_input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    print(f"Бот: {response}")

    chat_history_ids = output_ids