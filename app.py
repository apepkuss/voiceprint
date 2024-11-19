import os

import torchaudio
from speechbrain.pretrained import SpeakerRecognition


def load_audio(file_path):
    """加载音频文件并返回波形和采样率。"""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except Exception as e:
        print(f"加载音频文件 {file_path} 时出错: {e}")
        return None, None


def verify_speaker(file1, file2):
    """
    比较两个音频文件以验证是否来自同一说话人。
    """
    try:
        recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )

        # 在处理之前验证文件是否存在
        if not os.path.exists(file1) or not os.path.exists(file2):
            print("一个或两个音频文件不存在！")
            return

        score, prediction = recognizer.verify_files(file1, file2)
        print(f"说话人验证分数: {score:.4f}")
        print(f"是否为同一说话人: {'是' if prediction else '否'}")
    except Exception as e:
        print(f"说话人验证过程中出错: {e}")


def enroll_speaker(reference_audio, embeddings_dir="embeddings/"):
    """
    通过参考音频创建嵌入向量来注册说话人。
    """
    try:
        if not os.path.exists(reference_audio):
            print("参考音频文件不存在！")
            return

        os.makedirs(embeddings_dir, exist_ok=True)
        recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )

        # 将numpy导入移至文件顶部
        import numpy as np

        embedding = recognizer.encode_batch(reference_audio).squeeze(0).cpu().numpy()
        speaker_id = os.path.splitext(os.path.basename(reference_audio))[0]
        embedding_file = os.path.join(embeddings_dir, f"{speaker_id}.npy")

        np.save(embedding_file, embedding)
        print(f"已将说话人 {speaker_id} 的嵌入向量保存至 {embedding_file}")
    except Exception as e:
        print(f"说话人注册过程中出错: {e}")


def main():
    print("说话人识别应用")
    print("1. 验证两个音频文件")
    print("2. 注册说话人")
    print("3. 退出")

    while True:
        choice = input("请选择一个选项: ")
        if choice == "1":
            file1 = input("请输入第一个音频文件的路径: ")
            file2 = input("请输入第二个音频文件的路径: ")
            verify_speaker(file1, file2)
        elif choice == "2":
            file_path = input("请输入用于说话人注册的参考音频路径: ")
            enroll_speaker(file_path)
        elif choice == "3":
            print("正在退出...")
            break
        else:
            print("无效的选择。请重试。")


if __name__ == "__main__":
    main()
