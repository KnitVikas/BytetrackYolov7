import numpy as np
from collections import deque
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
from .kalman_filter import KalmanFilter
from yolox.tracker import matching
# STrack Class with Feature Storage
class STrack:
    shared_kalman = None  # Kalman filter instance

    def __init__(self, tlwh, score, feature=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.features = deque([], maxlen=100)  # Store last 100 features
        
        if feature is not None:
            self.features.append(feature)

    def update(self, new_track, frame_id, feature=None):
        """
        Update a matched track and append the new feature.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track._tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = 'Tracked'
        self.is_activated = True
        self.score = new_track.score
        
        # Append the new feature vector to the deque
        if feature is not None:
            self.features.append(feature)

    def smooth_feature(self):
        """
        Get the average feature vector from the deque.
        """
        return np.mean(self.features, axis=0) if len(self.features) > 0 else None

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """
        Convert bounding box from (top-left width-height) to (center x, center y, aspect ratio, height).
        """
        x, y, w, h = tlwh
        cx = x + w / 2
        cy = y + h / 2
        aspect_ratio = w / float(h)
        return np.array([cx, cy, aspect_ratio, h], dtype=np.float32)
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert bounding box from (min x, min y, max x, max y) 
        to (top left x, top left y, width, height).
        """
        print("tlbr_to_tlwh",tlbr)
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]  # Calculate width and height
        print(ret)
        return ret
# Feature Extractor Class
class FeatureExtractor:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # Load a pre-trained ResNet model for feature extraction
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Sequential() 
        # self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove the final classification layer
        self.model.eval()
        self.device = device
        self.model.to(device)
        # print(self.model)
        
        # Define image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        cv2.imwrite("crop.png",image)
        image = Image.fromarray(image)
       
        image = self.transform(image).unsqueeze(0).to(self.device)  # Preprocess image and add batch dimension
        with torch.no_grad():
            feature = self.model(image).squeeze().cpu().numpy()  # Extract features and convert to numpy
            # print("feature",feature.shape,type(feature))
        return feature


# Matching Utility Functions
def iou_distance(tracks, detections):
    """
    Compute IoU-based distance between tracks and detections.
    """
    # Placeholder for actual IoU computation logic
    return np.random.rand(len(tracks), len(detections))  # Random distances for demonstration

def feature_distance(tracks, detections):
    """
    Compute cosine distance between track features and detection features.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    track_features = np.array([track.smooth_feature() for track in tracks])
    detection_features = np.array([det.smooth_feature() for det in detections])

    # Normalize features
    track_features = track_features / np.linalg.norm(track_features, axis=1, keepdims=True)
    detection_features = detection_features / np.linalg.norm(detection_features, axis=1, keepdims=True)

    # Compute cosine similarity and convert to distance
    similarity = np.dot(track_features, detection_features.T)
    return 1 - similarity  # Distance is 1 - similarity

def linear_assignment(dists, thresh):
    """
    Assign detections to tracks using the Hungarian algorithm.
    """
    # Placeholder for Hungarian algorithm logic
    return [], [], []  # Placeholder for matches, unmatched tracks, and unmatched detections

# BYTETracker Class with Feature-Based Matching
# BYTETracker Class with Feature-Based Matching
class BYTETracker:
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # List of active STracks
        self.lost_stracks = []      # List of lost STracks
        self.removed_stracks = []   # List of removed STracks
        
        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.feature_extractor = FeatureExtractor()  # Initialize the feature extractor

    def update(self, output_results, img_info, img_size, img):
        """
        Update the tracker with new detections, considering both IoU and feature distances.
        """

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        #Extract bounding boxes and scores from the output
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]  # x1, y1, x2, y2 format

        # Rescale bboxes based on the image size
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # Extract features for the detections
        detection_features = []
        if len(dets) > 0:
            for bbox in dets:
                feature = self.feature_extractor.extract(
                    img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                )
                detection_features.append(feature)

            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, feature) 
                for (tlbr, s, feature) in zip(dets, scores_keep, detection_features)
            ]

        else:
            detections = []
       
        # Matching logic
        strack_pool = self.tracked_stracks + self.lost_stracks
        if len(strack_pool) == 0:
            dists = np.empty((0, 0))
        else:
            dists = iou_distance(strack_pool, detections)
            feature_dists = feature_distance(strack_pool, detections)
            dists = 0.5 * dists + 0.5 * feature_dists  # Weighted fusion
            print("dists : ", dists)

        # Use the matching procedure
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)
        print(" matches, u_track, u_detection " , matches, u_track, u_detection )

        # Handle matched tracks and detections
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            track.update(det, self.frame_id, det.features[-1])  # Update with new feature
            activated_starcks.append(track)

        # Handle unmatched tracks
        for it in u_track:
            track = strack_pool[it]
            if track.state == 'Tracked':
                track.mark_lost()
                lost_stracks.append(track)

        # Handle unmatched detections
        for idet in u_detection:
            track = detections[idet]
            if track.score >= self.det_thresh:  # Only keep high-confidence detections
                track.is_activated = False  # Mark as not activated
                self.lost_stracks.append(track)  # Add to lost stracks

        # Update tracked and lost lists
        self.tracked_stracks = [t for t in activated_starcks if t.is_activated]
        self.lost_stracks = [t for t in self.lost_stracks if t.tracklet_len <= self.max_time_lost]
        print( self.tracked_stracks,self.lost_stracks)

        # Remove lost tracks
        self.removed_stracks.extend([t for t in self.lost_stracks if t.tracklet_len > self.max_time_lost])
        self.lost_stracks = [t for t in self.lost_stracks if t.tracklet_len <= self.max_time_lost]
        
        # Debugging print statements
        print(f"Frame {self.frame_id}:")
        print(f"Tracked: {len(self.tracked_stracks)} | Lost: {len(self.lost_stracks)} | Removed: {len(self.removed_stracks)}")
        
        return self.tracked_stracks

