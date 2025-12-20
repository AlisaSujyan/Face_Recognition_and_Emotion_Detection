import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import pickle
from PIL import Image as PILImage
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

MODELS_DIR = Path(
    r"C:\Users\MSI GF66\OneDrive - National Polyechnic University of Armenia\Desktop\Emotion_recognition\models")
PROCESSED_DIR = Path(
    r"C:\Users\MSI GF66\OneDrive - National Polyechnic University of Armenia\Desktop\Emotion_recognition\data\processed")

EMOTION_MODEL_PATHS = [
    MODELS_DIR / f'enhanced_model_{i}.pth' for i in range(5)
]
FACE_MODEL_PATH = MODELS_DIR / 'ultimate_face_best.pth'

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.skip(x)
        out = F.relu(out)
        return out


class EmotionNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super(EmotionNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(torch.tensor(self.margin)) - sine * torch.sin(torch.tensor(self.margin))
        phi = torch.where(cosine > torch.cos(torch.tensor(self.margin)), phi,
                          cosine - torch.sin(torch.tensor(self.margin)) * self.margin)

        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class UltimateFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(UltimateFaceNet, self).__init__()

        resnet = models.resnet34(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.se = SEBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_global = nn.AdaptiveMaxPool2d((1, 1))

        self.bn_neck = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)

        self.fc_embedding = nn.Linear(1024, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

        self.arcface = ArcMarginProduct(embedding_dim, num_classes, scale=30.0, margin=0.50)

    def forward(self, x, labels=None, return_embedding=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.se(x)

        avg_pool = self.avgpool(x)
        max_pool = self.maxpool_global(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = torch.flatten(x, 1)

        x = self.bn_neck(x)
        x = self.dropout(x)

        embeddings = self.fc_embedding(x)
        embeddings = self.bn_embedding(embeddings)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        if return_embedding:
            return embeddings_norm

        output = self.arcface(embeddings_norm, labels)
        return output, embeddings_norm


class FaceEmotionInference:
    def __init__(self):
        print("Loading models...")

        with open(PROCESSED_DIR / 'faces' / 'label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)

        self.num_classes = label_mapping['num_classes']
        self.id_to_person = label_mapping['id_to_person']

        self.emotion_models = []
        for i, model_path in enumerate(EMOTION_MODEL_PATHS):
            if model_path.exists():
                model = EmotionNet(num_classes=7, dropout=0.5).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                self.emotion_models.append(model)
                print(f"  Loaded emotion model {i + 1}/5")

        self.face_model = UltimateFaceNet(num_classes=self.num_classes, embedding_dim=512).to(device)
        checkpoint = torch.load(FACE_MODEL_PATH)
        self.face_model.load_state_dict(checkpoint['model_state_dict'])
        self.face_model.eval()
        print(f"  Loaded face recognition model")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.emotion_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        print("Models loaded successfully!\n")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        return faces

    def predict_emotion(self, face_img):
        face_resized = cv2.resize(face_img, (48, 48))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        face_pil = PILImage.fromarray(face_gray, mode='L')
        face_tensor = self.emotion_transform(face_pil).unsqueeze(0).to(device)

        predictions = []
        with torch.no_grad():
            for model in self.emotion_models:
                output = model(face_tensor)
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

        avg_probs = np.mean(predictions, axis=0)
        emotion_idx = np.argmax(avg_probs)
        confidence = avg_probs[0][emotion_idx]

        return EMOTION_LABELS[emotion_idx], confidence

    def predict_face(self, face_img):
        face_resized = cv2.resize(face_img, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        face_pil = PILImage.fromarray(face_rgb)
        face_tensor = self.face_transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embeddings = self.face_model(face_tensor, return_embedding=True)
            cos_sim = F.linear(embeddings, F.normalize(self.face_model.arcface.weight))
            probs = F.softmax(cos_sim, dim=1)

            top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)

            person_id = top5_indices[0][0].item()
            confidence = top5_probs[0][0].item()

        person_name = self.id_to_person[person_id].replace('_', ' ')

        return person_name, confidence

    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        results = []

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]

            emotion, emotion_conf = self.predict_emotion(face_img)
            person, person_conf = self.predict_face(face_img)

            results.append({
                'bbox': (x, y, w, h),
                'person': person,
                'person_confidence': person_conf,
                'emotion': emotion,
                'emotion_confidence': emotion_conf
            })

        return results

    def draw_results(self, frame, results, show_person=True, show_emotion=True):
        """
        Draw bounding boxes and labels on frame

        Args:
            frame: Input frame
            results: Detection results
            show_person: Whether to show person name label
            show_emotion: Whether to show emotion label
        """
        for result in results:
            x, y, w, h = result['bbox']
            person = result['person']
            person_conf = result['person_confidence']
            emotion = result['emotion']
            emotion_conf = result['emotion_confidence']

            color = (0, 255, 0) if person_conf > 0.7 else (0, 165, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            labels = []
            label_sizes = []

            # Build labels based on parameters
            if show_person:
                label_person = f"{person} ({person_conf * 100:.1f}%)"
                labels.append(label_person)
                label_size_person, _ = cv2.getTextSize(label_person, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_sizes.append(label_size_person[0])

            if show_emotion:
                label_emotion = f"{emotion.capitalize()} ({emotion_conf * 100:.1f}%)"
                labels.append(label_emotion)
                label_size_emotion, _ = cv2.getTextSize(label_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_sizes.append(label_size_emotion[0])

            if labels:
                # Calculate background rectangle size
                max_width = max(label_sizes)
                rect_height = 25 * len(labels) + 5

                cv2.rectangle(frame, (x, y - rect_height), (x + max_width + 10, y), (0, 0, 0), -1)

                # Draw labels
                y_offset = -rect_height + 20
                for label in labels:
                    cv2.putText(frame, label, (x + 5, y + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25

        return frame

    def run_webcam(self):
        print("Starting webcam...")
        print("Press 'q' to quit, 's' to save screenshot")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_time = time.time()
        fps = 0
        screenshot_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            results = self.process_frame(frame)

            # Show only emotion label for webcam
            frame = self.draw_results(frame, results, show_person=False, show_emotion=True)

            current_time = time.time()
            fps = 1 / (current_time - fps_time)
            fps_time = current_time

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(results)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Face Recognition & Emotion Detection', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")

    def process_image(self, image_path, output_path=None):
        print(f"Processing image: {image_path}")

        frame = cv2.imread(image_path)

        if frame is None:
            print("Error: Could not read image")
            return

        results = self.process_frame(frame)

        # Show both person and emotion labels for images
        frame = self.draw_results(frame, results, show_person=True, show_emotion=True)

        print(f"\nDetected {len(results)} face(s):")
        for i, result in enumerate(results, 1):
            print(f"\nFace {i}:")
            print(f"  Person: {result['person']} ({result['person_confidence'] * 100:.1f}%)")
            print(f"  Emotion: {result['emotion'].capitalize()} ({result['emotion_confidence'] * 100:.1f}%)")

        if output_path is None:
            output_path = image_path.replace('.', '_result.')

        cv2.imwrite(output_path, frame)
        print(f"\nResult saved to: {output_path}")

        cv2.imshow('Result', frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path, output_path=None):
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            output_path = video_path.replace('.', '_result.')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print(f"Processing {total_frames} frames...")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = self.process_frame(frame)

            # Show only emotion label for video
            frame = self.draw_results(frame, results, show_person=False, show_emotion=True)

            out.write(frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)")

        cap.release()
        out.release()

        print(f"\nVideo processing complete!")
        print(f"Result saved to: {output_path}")


def main():
    print("=" * 70)
    print("FACE RECOGNITION & EMOTION DETECTION SYSTEM")
    print("=" * 70)

    inference = FaceEmotionInference()

    while True:
        print("\n" + "=" * 70)
        print("SELECT MODE:")
        print("=" * 70)
        print("1. Webcam (Real-time)")
        print("2. Image File")
        print("3. Video File")
        print("4. Exit")
        print("=" * 70)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            inference.run_webcam()

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            output_path = input("Enter output path (or press Enter for default): ").strip()

            if not output_path:
                output_path = None

            inference.process_image(image_path, output_path)

        elif choice == '3':
            video_path = input("Enter video path: ").strip()
            output_path = input("Enter output path (or press Enter for default): ").strip()

            if not output_path:
                output_path = None

            inference.process_video(video_path, output_path)

        elif choice == '4':
            print("\nExiting... Goodbye!")
            break

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()