import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';
import '@mediapipe/drawing_utils';

import { vehicleService } from '@/services/supabaseVehicleService';
import { useToast } from '@/hooks/use-toast';
import { AlertTriangle, Eye, Camera as CamOn, CameraOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface Props {
  vehicleId: string;
  isActive: boolean;
  onDetection?: (state: DrowsinessState, ear: number) => void;
}

type DrowsinessState = 'awake' | 'drowsy' | 'sleeping';

/* ────────────────────────────────────
 * Landmark indices (MediaPipe FaceMesh)
 * ──────────────────────────────────── */
const LEFT_EYE_POINTS = [
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
];
const RIGHT_EYE_POINTS = [
  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
];

const LEFT_EYE_CORNERS = [33, 133];  // inner, outer
const RIGHT_EYE_CORNERS = [362, 263];
const LEFT_EYE_VERTICAL = [159, 145, 158, 153];   // top1, bottom1, top2, bottom2
const RIGHT_EYE_VERTICAL = [386, 374, 387, 373];

/* ──────────────────────────────────── */

const MediaPipeDrowsinessDetector: React.FC<Props> = ({
  vehicleId,
  isActive,
  onDetection,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  /* refs that change every frame */
  const earRef            = useRef(0);
  const closedFramesRef   = useRef(0);
  const drowsyFramesRef   = useRef(0);
  const currentStateRef   = useRef<DrowsinessState>('awake');
  const lastWriteRef      = useRef(0);

  /* UI state (updates a few times / s) */
  const [eyeAspectRatio, setEyeAspectRatio] = useState(0);
  const [stateUI,        setStateUI]        = useState<DrowsinessState>('awake');
  const [faceFound,      setFaceFound]      = useState(false);
  const [hasCamera,      setHasCamera]      = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const { toast } = useToast();

  /* ───────────── EAR helpers ───────────── */
  const dist = (a: any, b: any) =>
    Math.hypot(a.x - b.x, a.y - b.y);

  const calcEAR = (lm: any[]): number => {
    try {
      const leftH  = dist(lm[LEFT_EYE_CORNERS[0]], lm[LEFT_EYE_CORNERS[1]]);
      const leftV1 = dist(lm[LEFT_EYE_VERTICAL[0]], lm[LEFT_EYE_VERTICAL[1]]);
      const leftV2 = dist(lm[LEFT_EYE_VERTICAL[2]], lm[LEFT_EYE_VERTICAL[3]]);
      const rightH  = dist(lm[RIGHT_EYE_CORNERS[0]], lm[RIGHT_EYE_CORNERS[1]]);
      const rightV1 = dist(lm[RIGHT_EYE_VERTICAL[0]], lm[RIGHT_EYE_VERTICAL[1]]);
      const rightV2 = dist(lm[RIGHT_EYE_VERTICAL[2]], lm[RIGHT_EYE_VERTICAL[3]]);

      return ((leftV1 + leftV2) / (2 * leftH) + (rightV1 + rightV2) / (2 * rightH)) / 2;
    } catch {
      return 0.25;  // assume eyes open on error
    }
  };

  /* ───────────── Audio alerts ───────────── */
  const playAlert = useCallback((kind: 'drowsy' | 'sleeping') => {
    try {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      if (kind === 'drowsy') {
        [0, 0.2, 0.4].forEach(delay => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.frequency.value = 500;
          osc.connect(gain);
          gain.connect(ctx.destination);
          gain.gain.setValueAtTime(0.3, ctx.currentTime + delay);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + 0.1);
          osc.start(ctx.currentTime + delay);
          osc.stop(ctx.currentTime + delay + 0.1);
        });
      } else {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = 1000;
        osc.connect(gain);
        gain.connect(ctx.destination);
        gain.gain.value = 0.5;
        osc.start();
        osc.stop(ctx.currentTime + 1.5);
      }
    } catch (e) {
      console.error('alert error', e);
    }
  }, []);

  /* ───────────── main state machine ───────────── */
  const advanceState = useCallback((ear: number) => {
    earRef.current = ear;

    let newState: DrowsinessState = 'awake';
    if (ear < 0.15) {                       // eyes shut
      closedFramesRef.current++;
      drowsyFramesRef.current = 0;
      if (closedFramesRef.current >= 22) newState = 'sleeping';
    } else if (ear < 0.20) {               // partially shut
      drowsyFramesRef.current++;
      closedFramesRef.current = 0;
      if (drowsyFramesRef.current >= 30) newState = 'drowsy';
    } else {                               // fully open
      closedFramesRef.current = 0;
      drowsyFramesRef.current = 0;
    }

    if (newState !== currentStateRef.current) {
      currentStateRef.current = newState;
      setStateUI(newState);

      /* throttled DB write */
      if (Date.now() - lastWriteRef.current > 5000) {
        const level = newState === 'sleeping' ? 'severe'
                    : newState === 'drowsy'  ? 'moderate'
                    : 'none';
        if (level !== 'none') {
          vehicleService.recordDrowsinessEvent({
            vehicle_id: vehicleId,
            drowsiness_level: level,
            confidence: 0.9,
            eye_aspect_ratio: ear,
            alert_triggered: newState !== 'drowsy',
          }).catch(console.error);
          lastWriteRef.current = Date.now();
        }
      }

      /* UI + sound */
      if (newState === 'drowsy') {
        playAlert('drowsy');
        toast({ title: '⚠️ DROWSINESS DETECTED', variant: 'destructive' });
      } else if (newState === 'sleeping') {
        playAlert('sleeping');
        toast({ title: '🚨 SLEEPING DETECTED', variant: 'destructive' });
      }

      onDetection?.(newState, ear);
    }
  }, [playAlert, toast, vehicleId, onDetection]);

  /* ───────────── Mediapipe initialisation ───────────── */
  const cameraRef = useRef<Camera | null>(null);
  const meshRef   = useRef<FaceMesh | null>(null);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      cameraRef.current?.stop();
      const mesh = new FaceMesh({
        locateFile: f => `/node_modules/@mediapipe/face_mesh/${f}`,
      });
      mesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5,
      });

      mesh.onResults(res => {
        const ok = !!res.multiFaceLandmarks?.length;
        setFaceFound(ok);
        if (ok) {
          const ear = calcEAR(res.multiFaceLandmarks![0]);
          advanceState(ear);
          drawEyeLandmarks(res.multiFaceLandmarks![0]);
        }
      });

      meshRef.current = mesh;

      const cam = new Camera(videoRef.current!, {
        onFrame: async () => {
          await mesh.send({ image: videoRef.current! });
        },
        width: 640,
        height: 480,
      });
      cameraRef.current = cam;
      cam.start();
      setHasCamera(true);
    } catch (e) {
      console.error(e);
      setError('Camera unavailable – check permissions & HTTPS.');
      setHasCamera(false);
    }
  }, [advanceState]);

  const stopCamera = useCallback(() => {
    cameraRef.current?.stop();
    cameraRef.current = null;
    meshRef.current?.reset();
    meshRef.current = null;
    setHasCamera(false);
    setFaceFound(false);
    currentStateRef.current = 'awake';
    setStateUI('awake');
  }, []);

  /* ───────────── draw helpers ───────────── */
  const drawEyeLandmarks = (lm: any[]) => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx || !canvasRef.current) return;
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    const drawSet = (pts: number[], color: string) => {
      ctx.fillStyle = color;
      pts.forEach(i => {
        ctx.beginPath();
        ctx.arc(lm[i].x * canvasRef.current!.width,
                lm[i].y * canvasRef.current!.height,
                2, 0, 2 * Math.PI);
        ctx.fill();
      });
    };
    drawSet(LEFT_EYE_POINTS,  '#00ff00');
    drawSet(RIGHT_EYE_POINTS, '#0000ff');
  };

  /* ───────────── expose EAR at 5 Hz ───────────── */
  useEffect(() => {
    const id = setInterval(() => setEyeAspectRatio(earRef.current), 200);
    return () => clearInterval(id);
  }, []);

  /* ───────────── handle isActive prop ───────────── */
  useEffect(() => {
    if (isActive && !hasCamera) startCamera();
    if (!isActive && hasCamera) stopCamera();
  }, [isActive, hasCamera, startCamera, stopCamera]);

  /* cleanup on unmount */
  useEffect(() => () => stopCamera(), [stopCamera]);

  /* ───────────── UI helpers ───────────── */
  const stateColor = (s: DrowsinessState) =>
    s === 'awake'    ? 'bg-green-500'
  : s === 'drowsy'   ? 'bg-yellow-500'
  : s === 'sleeping' ? 'bg-red-500'
  : 'bg-gray-500';

  /* ───────────── JSX ───────────── */
  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" /> Driver Drowsiness Detector
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="relative">
          <video
            ref={videoRef}
            className="w-full rounded-lg bg-gray-100"
            width={320}
            height={240}
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            width={320}
            height={240}
          />

          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
              <div className="text-center p-4">
                <CameraOff className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                <p className="text-sm text-gray-600">{error}</p>
              </div>
            </div>
          )}

          {hasCamera && (
            <div className="absolute top-2 right-2 flex gap-2">
              <div className={`w-3 h-3 rounded-full ${faceFound ? 'bg-green-500' : 'bg-red-500'}`} />
              {isActive && <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />}
            </div>
          )}
        </div>

        <div className="flex gap-2">
          {!hasCamera ? (
            <Button onClick={startCamera} className="flex-1">
              <CamOn className="mr-2 h-4 w-4" />
              Start Camera
            </Button>
          ) : (
            <Button onClick={stopCamera} variant="outline" className="flex-1">
              <CameraOff className="mr-2 h-4 w-4" />
              Stop Camera
            </Button>
          )}
        </div>

        {hasCamera && (
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm font-medium">State:</span>
              <Badge className={stateColor(stateUI)}>{stateUI.toUpperCase()}</Badge>
            </div>

            <div className="flex justify-between">
              <span className="text-sm font-medium">Face:</span>
              <span className={`text-sm ${faceFound ? 'text-green-600' : 'text-red-600'}`}>
                {faceFound ? 'Detected' : 'Not Found'}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-sm font-medium">Eye Ratio:</span>
              <span className="text-sm font-mono">{eyeAspectRatio.toFixed(4)}</span>
            </div>

            {stateUI !== 'awake' && (
              <div className="flex items-center gap-2 p-2 bg-yellow-50 rounded border border-yellow-200">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                <span className="text-sm text-yellow-800">
                  {stateUI === 'sleeping' ? 'WAKE UP! Pull over now!' : 'Stay alert while driving.'}
                </span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default MediaPipeDrowsinessDetector;
