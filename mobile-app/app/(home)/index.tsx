import { useEffect, useMemo, useRef, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Platform,
  Pressable,
} from "react-native";
import Svg, { Rect, Text as SvgText, G } from "react-native-svg";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
  useFrameProcessor,
} from "react-native-vision-camera";
import { useRunOnJS, useSharedValue } from "react-native-worklets-core";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import DeviceInfo from "react-native-device-info";

const LABELS = [
  "boots",
  "glasses",
  "gloves",
  "helmet",
  "no boots",
  "no glasses",
  "no gloves",
  "no helmet",
  "no vest",
  "person",
  "vest",
];

const INPUT_SIZE = 320;
const NUM_CLASSES = 11;
const MAX_DETECTIONS = 300;
const INFERENCE_INTERVAL_MS = 0; // Run at maximum possible hardware speed
const TARGET_OVERLAY_FPS = 30;
const OVERLAY_INTERVAL_MS = 1000 / TARGET_OVERLAY_FPS;
const RAM_UPDATE_INTERVAL_MS = 500;
const MAX_RENDER_DETS = 5;
const CAMERA_FPS = 30;
const BYTES_PER_MB = 1024 * 1024;

const CONF_THR = 0.50;
const IOU_THR = 0.45;

type Det = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls: number;
  label: string;
};

type FrameOrientation =
  | "portrait"
  | "portrait-upside-down"
  | "landscape-left"
  | "landscape-right"
  | string;

function parseBytes(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return value;
  }
  if (typeof value === "bigint") {
    const next = Number(value);
    return Number.isFinite(next) && next > 0 ? next : null;
  }
  if (typeof value !== "string") return null;

  const normalized = value.trim().toLowerCase().replace(/,/g, "");
  if (!normalized) return null;

  const match = normalized.match(/([-+]?\d*\.?\d+)\s*(tb|gb|mb|kb|b)?/);
  if (!match) return null;

  const numeric = Number(match[1]);
  if (!Number.isFinite(numeric) || numeric <= 0) return null;

  const unit = match[2] ?? "b";
  if (unit === "kb") return numeric * 1024;
  if (unit === "mb") return numeric * BYTES_PER_MB;
  if (unit === "gb") return numeric * BYTES_PER_MB * 1024;
  if (unit === "tb") return numeric * BYTES_PER_MB * 1024 * 1024;
  return numeric;
}

function getRuntimeMemoryMB(): number | null {
  const perf = (globalThis as any)?.performance;
  const perfHeap = perf?.memory?.usedJSHeapSize;
  if (typeof perfHeap === "number" && Number.isFinite(perfHeap) && perfHeap > 0) {
    return Math.round(perfHeap / BYTES_PER_MB);
  }

  const hermes = (globalThis as any)?.HermesInternal;
  if (!hermes?.getRuntimeProperties) return null;

  let props: Record<string, unknown> | null = null;
  try {
    props = hermes.getRuntimeProperties();
  } catch {
    return null;
  }
  if (!props || typeof props !== "object") return null;

  let bestBytes = 0;
  for (const [key, rawValue] of Object.entries(props)) {
    const lowKey = key.toLowerCase();
    if (!/(heap|memory|alloc|usage|working|bytes)/.test(lowKey)) continue;

    const bytes = parseBytes(rawValue);
    if (bytes !== null && bytes > bestBytes) {
      bestBytes = bytes;
    }
  }

  return bestBytes > 0 ? Math.round(bestBytes / BYTES_PER_MB) : null;
}

function sigmoid(x: number) {
  "worklet";
  return 1 / (1 + Math.exp(-x));
}

function normalizeProbability(raw: number) {
  "worklet";
  if (!Number.isFinite(raw)) return 0;
  if (raw < 0 || raw > 1) return sigmoid(raw);
  return raw;
}

function iou(a: Det, b: Det) {
  "worklet";
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nms(boxes: Det[], iouThr: number) {
  "worklet";
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep: Det[] = [];

  for (const b of sorted) {
    let ok = true;
    for (const k of keep) {
      if (iou(b, k) > iouThr) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(b);
  }

  return keep;
}

function toFloatArray(data: ArrayLike<number | bigint>): Float32Array {
  "worklet";
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i];
    out[i] = typeof value === "bigint" ? Number(value) : value;
  }
  return out;
}

function mapPointToUpright(
  x: number,
  y: number,
  frameW: number,
  frameH: number,
  orientation: FrameOrientation,
): { x: number; y: number } {
  if (orientation === "landscape-left") {
    return { x: frameH - y, y: x };
  }
  if (orientation === "landscape-right") {
    return { x: y, y: frameW - x };
  }
  if (orientation === "portrait-upside-down") {
    return { x: frameW - x, y: frameH - y };
  }
  return { x, y };
}

function getUprightFrameSize(
  frameW: number,
  frameH: number,
  orientation: FrameOrientation,
): { width: number; height: number } {
  if (orientation === "landscape-left" || orientation === "landscape-right") {
    return { width: frameH, height: frameW };
  }
  return { width: frameW, height: frameH };
}

// Output shape: [1, 300, 6] — end2end model with built-in NMS
// Each row: [x1, y1, x2, y2, confidence, class_id] (normalized 0–1 coords)
function postprocess(output: Float32Array): Det[] {
  "worklet";
  const dets: Det[] = [];

  for (let i = 0; i < MAX_DETECTIONS; i += 1) {
    const base = i * 6;
    const x1 = output[base];
    const y1 = output[base + 1];
    const x2 = output[base + 2];
    const y2 = output[base + 3];
    const conf = output[base + 4];
    const cls = Math.round(output[base + 5]);

    if (conf < CONF_THR) continue;
    if (!Number.isFinite(x1) || !Number.isFinite(y1)) continue;

    // Coords are normalized (0–1), scale to pixel space [0..INPUT_SIZE]
    dets.push({
      x1: x1 * INPUT_SIZE,
      y1: y1 * INPUT_SIZE,
      x2: x2 * INPUT_SIZE,
      y2: y2 * INPUT_SIZE,
      score: conf,
      cls,
      label: LABELS[cls] ?? `cls_${cls}`,
    });
  }

  return dets;
}

function mapToScreen(
  dets: Det[],
  screenW: number,
  screenH: number,
  frameW: number,
  frameH: number,
  orientation: FrameOrientation,
): Det[] {
  if (frameW <= 0 || frameH <= 0) return [];

  const isSquareFrame = frameW === frameH;
  // If the camera stream is already square, avoid extra crop calculations.
  const cropSize = isSquareFrame ? frameW : Math.min(frameW, frameH);
  const cropX = isSquareFrame ? 0 : (frameW - cropSize) / 2;
  const cropY = isSquareFrame ? 0 : (frameH - cropSize) / 2;

  const uprightSize = getUprightFrameSize(frameW, frameH, orientation);
  const previewScale = Math.max(
    screenW / uprightSize.width,
    screenH / uprightSize.height,
  );
  const previewOffsetX = (screenW - uprightSize.width * previewScale) / 2;
  const previewOffsetY = (screenH - uprightSize.height * previewScale) / 2;

  return dets.map((d) => {
    // Model-space (0..INPUT_SIZE) -> source frame coordinates.
    const fx1 = cropX + (d.x1 / INPUT_SIZE) * cropSize;
    const fy1 = cropY + (d.y1 / INPUT_SIZE) * cropSize;
    const fx2 = cropX + (d.x2 / INPUT_SIZE) * cropSize;
    const fy2 = cropY + (d.y2 / INPUT_SIZE) * cropSize;

    // Rotate into upright preview coordinates.
    const p1 = mapPointToUpright(fx1, fy1, frameW, frameH, orientation);
    const p2 = mapPointToUpright(fx2, fy1, frameW, frameH, orientation);
    const p3 = mapPointToUpright(fx2, fy2, frameW, frameH, orientation);
    const p4 = mapPointToUpright(fx1, fy2, frameW, frameH, orientation);

    const ux1 = Math.min(p1.x, p2.x, p3.x, p4.x);
    const uy1 = Math.min(p1.y, p2.y, p3.y, p4.y);
    const ux2 = Math.max(p1.x, p2.x, p3.x, p4.x);
    const uy2 = Math.max(p1.y, p2.y, p3.y, p4.y);

    const rawX1 = ux1 * previewScale + previewOffsetX;
    const rawY1 = uy1 * previewScale + previewOffsetY;
    const rawX2 = ux2 * previewScale + previewOffsetX;
    const rawY2 = uy2 * previewScale + previewOffsetY;

    const clampedX1 = Math.max(0, Math.min(screenW, rawX1));
    const clampedY1 = Math.max(0, Math.min(screenH, rawY1));
    const clampedX2 = Math.max(0, Math.min(screenW, rawX2));
    const clampedY2 = Math.max(0, Math.min(screenH, rawY2));

    return {
      ...d,
      x1: Math.min(clampedX1, clampedX2),
      y1: Math.min(clampedY1, clampedY2),
      x2: Math.max(clampedX1, clampedX2),
      y2: Math.max(clampedY1, clampedY2),
    };
  });
}

function limitDetections(dets: Det[], maxCount: number): Det[] {
  "worklet";
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const nonPerson: Det[] = [];
  const persons: Det[] = [];

  for (const det of sorted) {
    if (det.label === "person") {
      persons.push(det);
    } else {
      nonPerson.push(det);
    }
  }

  return [...nonPerson, ...persons].slice(0, maxCount);
}

function toImageUri(pathOrUri: string): string {
  if (pathOrUri.startsWith("file://")) return pathOrUri;
  if (pathOrUri.startsWith("content://")) return pathOrUri;
  return `file://${pathOrUri}`;
}

export default function LiveDetectScreen() {
  const router = useRouter();
  const cameraRef = useRef<Camera>(null);
  const [cameraFacing, setCameraFacing] = useState<"back" | "front">("back");
  const backDevice = useCameraDevice("back");
  const frontDevice = useCameraDevice("front");
  const device = cameraFacing === "back" ? backDevice : frontDevice;
  const format = useCameraFormat(device, [
    { videoAspectRatio: 1 },
    { videoResolution: { width: INPUT_SIZE, height: INPUT_SIZE } },
    { fps: CAMERA_FPS },
  ]);
  const { hasPermission, requestPermission } = useCameraPermission();
  const { resize } = useResizePlugin();

  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [dets, setDets] = useState<Det[]>([]);
  const [isNavigating, setIsNavigating] = useState(false);

  // Performance Stats
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [ram, setRam] = useState<number | null>(null);

  const canFlipCamera = Boolean(backDevice && frontDevice);
  const lastRunMs = useSharedValue(0);
  const lastOverlayMs = useSharedValue(0);

  const { width: screenW, height: screenH } = Dimensions.get("window");
  const sendDetectionsToJs = useRunOnJS(
    (
      next: Det[],
      frameW: number,
      frameH: number,
      orientation: FrameOrientation,
      stats: { fps: number; latency: number }
    ) => {
      setDets(mapToScreen(next, screenW, screenH, frameW, frameH, orientation));
      setFps(stats.fps);
      setLatency(stats.latency);
    },
    [screenW, screenH],
  );

  useEffect(() => {
    (async () => {
      const loaded = await loadTensorflowModel(
        require("../../assets/models/best_float16 (2).tflite"),
        "gpu" as any
      );
      setModel(loaded);
    })();
  }, []);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    let mounted = true;

    const updateMemory = async () => {
      let nextRam: number | null = null;

      try {
        const usedBytes = await DeviceInfo.getUsedMemory();
        if (Number.isFinite(usedBytes) && usedBytes > 0) {
          nextRam = Math.round(usedBytes / BYTES_PER_MB);
        }
      } catch {
        // Ignore and try runtime fallback.
      }

      if (nextRam === null) {
        nextRam = getRuntimeMemoryMB();
      }

      if (mounted && nextRam !== null) {
        setRam(nextRam);
      }
    };

    updateMemory();
    const timer = setInterval(updateMemory, RAM_UPDATE_INTERVAL_MS);

    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (cameraFacing === "back" && !backDevice && frontDevice) {
      setCameraFacing("front");
      return;
    }
    if (cameraFacing === "front" && !frontDevice && backDevice) {
      setCameraFacing("back");
    }
  }, [cameraFacing, backDevice, frontDevice]);

  const ready = useMemo(() => model !== null, [model]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      if (!model) return;

      const start = performance.now();

      // Keep frame processing bounded only by hardware.
      const now = performance.now();
      if (INFERENCE_INTERVAL_MS > 0 && now - lastRunMs.value < INFERENCE_INTERVAL_MS) return;
      lastRunMs.value = now;

      const input = resize(frame, {
        scale: { width: INPUT_SIZE, height: INPUT_SIZE },
        pixelFormat: "rgb",
        dataType: "float32",
      });

      const outputs = model.runSync([input]);
      const end = performance.now();
      const currentLatency = end - start;

      // Calculate stats even if no detections
      const overlayDiff = now - lastOverlayMs.value;
      if (overlayDiff >= OVERLAY_INTERVAL_MS) {
        const currentFps = 1000 / overlayDiff;
        lastOverlayMs.value = now;

        const parsed = outputs.length > 0 ? postprocess(toFloatArray(outputs[0])) : [];
        const limited = limitDetections(parsed, MAX_RENDER_DETS);

        sendDetectionsToJs(
          limited,
          frame.width,
          frame.height,
          frame.orientation,
          {
            fps: Math.round(currentFps),
            latency: Math.round(currentLatency),
          }
        );
      }
    },
    [model, resize, lastRunMs, lastOverlayMs, sendDetectionsToJs],
  );

  const goToResult = (imageUri: string) => {
    router.push({
      pathname: "/(home)/result",
      params: {
        imageUri,
      },
    });
  };

  const onToggleFacing = () => {
    if (!canFlipCamera) return;
    setCameraFacing((prev) => (prev === "back" ? "front" : "back"));
  };

  const onCapture = async () => {
    if (!cameraRef.current || isNavigating) return;
    setIsNavigating(true);
    try {
      const photo = await cameraRef.current.takePhoto();
      goToResult(toImageUri(photo.path));
    } catch (error) {
      console.warn("capture_failed", error);
    } finally {
      setIsNavigating(false);
    }
  };

  const onPickImage = async () => {
    if (isNavigating) return;
    setIsNavigating(true);
    try {
      const permission =
        await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) return;

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"],
        allowsEditing: false,
        quality: 1,
      });
      if (result.canceled || !result.assets.length) return;

      goToResult(result.assets[0].uri);
    } catch (error) {
      console.warn("pick_image_failed", error);
    } finally {
      setIsNavigating(false);
    }
  };

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>No camera device found.</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>Camera permission is required.</Text>
      </View>
    );
  }

  if (!ready) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>Loading model...</Text>
      </View>
    );
  }

  if (!format) {
    return (
      <View style={styles.center}>
        <Text style={styles.txt}>Preparing camera format...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        photo={true}
        format={format}
        fps={CAMERA_FPS}
      />

      <View style={styles.topPill}>
        <View style={styles.dot} />
        <Text style={styles.topPillText}>LIVE SCAN ACTIVE</Text>
      </View>

      <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
        {dets.map((d, idx) => {
          const w = d.x2 - d.x1;
          const h = d.y2 - d.y1;
          if (w < 2 || h < 2) return null;
          return (
            <G key={idx}>
              <Rect
                x={d.x1}
                y={d.y1}
                width={w}
                height={h}
                stroke="lime"
                strokeWidth={2}
                fill="transparent"
                rx={6}
                ry={6}
              />
              <Rect
                x={Math.max(0, Math.min(screenW - 180, d.x1))}
                y={Math.max(0, d.y1 - 22)}
                width={180}
                height={20}
                fill="rgba(0,0,0,0.55)"
                rx={6}
                ry={6}
              />
              <SvgText
                x={Math.max(8, Math.min(screenW - 172, d.x1 + 8))}
                y={Math.max(14, d.y1 - 8)}
                fill="white"
                fontSize={12}
                fontWeight="700"
              >
                {d.label} {(d.score * 100).toFixed(0)}%
              </SvgText>
            </G>
          );
        })}
      </Svg>

      <View style={styles.statsOverlay}>
        <Text style={styles.statsText}>FPS: {fps}</Text>
        <Text style={styles.statsText}>Latency: {latency}ms</Text>
        <Text style={styles.statsText}>RAM: {ram !== null ? `${ram}MB` : "N/A"}</Text>
      </View>

      <View style={styles.controlsWrap}>
        <Pressable
          style={[styles.pickButton, isNavigating && styles.btnDisabled]}
          onPress={onPickImage}
          disabled={isNavigating}
        >
          <Text style={styles.pickButtonText}>Pick</Text>
        </Pressable>

        <Pressable
          style={[styles.captureOuter, isNavigating && styles.btnDisabled]}
          onPress={onCapture}
          disabled={isNavigating}
        >
          <View style={styles.captureInner} />
        </Pressable>

        <Pressable
          style={[
            styles.facingButton,
            (!canFlipCamera || isNavigating) && styles.btnDisabled,
          ]}
          onPress={onToggleFacing}
          disabled={!canFlipCamera || isNavigating}
        >
          <Text style={styles.facingButtonText}>Flip</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "black" },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "black",
  },
  txt: { color: "white" },

  topPill: {
    position: "absolute",
    top: Platform.select({ ios: 60, android: 40, default: 40 }),
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 8,
    backgroundColor: "lime",
    marginRight: 8,
  },
  topPillText: {
    color: "white",
    fontWeight: "800",
    letterSpacing: 0.6,
    fontSize: 12,
  },
  controlsWrap: {
    position: "absolute",
    bottom: Platform.select({ ios: 30, android: 24, default: 24 }),
    left: 0,
    right: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 26,
  },
  pickButton: {
    width: 62,
    height: 62,
    borderRadius: 31,
    backgroundColor: "rgba(0,0,0,0.7)",
    borderWidth: 2,
    borderColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  pickButtonText: {
    color: "white",
    fontSize: 13,
    fontWeight: "700",
  },
  captureOuter: {
    width: 82,
    height: 82,
    borderRadius: 41,
    borderWidth: 4,
    borderColor: "white",
    backgroundColor: "rgba(255,255,255,0.2)",
    alignItems: "center",
    justifyContent: "center",
  },
  captureInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "white",
  },
  facingButton: {
    width: 62,
    height: 62,
    borderRadius: 31,
    backgroundColor: "rgba(0,0,0,0.7)",
    borderWidth: 2,
    borderColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  facingButtonText: {
    color: "white",
    fontSize: 13,
    fontWeight: "700",
  },
  btnDisabled: {
    opacity: 0.6,
  },
  statsOverlay: {
    position: "absolute",
    top: Platform.select({ ios: 60, android: 45, default: 45 }),
    right: 20,
    backgroundColor: "rgba(0,0,0,0.6)",
    padding: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  statsText: {
    color: "#00ff00",
    fontSize: 11,
    fontWeight: "bold",
    fontFamily: Platform.select({ ios: "Courier", android: "monospace" }),
  },
});
