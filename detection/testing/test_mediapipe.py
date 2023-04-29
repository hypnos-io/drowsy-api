import time

from detection.head_detector import HeadDetector, create_frame_list

head = HeadDetector()

start_time = time.time()

frame_sequence = create_frame_list("png")

current_time = time.time() - start_time
print(f"Tempo pra criar a lista de frames: {current_time} segundos\n")

start_time = time.time()

result_head = head.execute(frame_sequence)
print("=-=-=-=-=-=-=-=-=")
print("\033[32mDetection Results: \033[0m\n")
for key, value in result_head.data.items():
    print(f"{key}: {value:.2f}")
print(f"\033[31mTotal Result: {result_head.result}\033[0m")
print("=-=-=-=-=-=-=-=-=")

current_time = time.time() - start_time
print(f"Tempo pra executar a detect: {current_time} segundos\n")

# classifier = KSSClassifier(len(frame_sequence), result_eye, result_head)
# response = classifier.classify(weights)
# print(response)
