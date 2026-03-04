import { useLocalSearchParams, useRouter } from "expo-router";
import { useEffect, useMemo, useState } from "react";
import { Dimensions, Image, Platform, Pressable, StyleSheet, Text, View, ScrollView } from "react-native";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import { loadTensorflowModel, TensorflowModel } from "react-native-fast-tflite";
import { SafeAreaView } from "react-native-safe-area-context";
import Svg, { Rect, Text as SvgText, G, Path } from "react-native-svg";
import { toByteArray } from "base64-js";
import * as jpeg from "jpeg-js";

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
const MAX_DETECTIONS = 300;
const CONF_THR = 0.25;

type Detection = {
  classId: number;
  className: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type ModelDet = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls: number;
  label: string;
};

function toTitleCase(value: string): string {
  return value
    .split(" ")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

function formatSummary(detections: Detection[]): string {
  if (!detections.length) return "No detections";
  const unique = Array.from(
    new Set(detections.map((d) => toTitleCase(d.className))),
  );
  return unique.join(", ");
}



function toFloatArray(data: ArrayLike<number | bigint>): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i];
    out[i] = typeof value === "bigint" ? Number(value) : value;
  }
  return out;
}

// Output shape: [1, 300, 6] — end2end model with built-in NMS
// Each row: [x1, y1, x2, y2, confidence, class_id] (normalized 0–1 coords)
function postprocess(output: Float32Array): ModelDet[] {
  const dets: ModelDet[] = [];

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

    // Coords are normalized (0–1), scale to pixel space
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

async function imageToTensor(imageUri: string): Promise<Float32Array> {
  const resized = await manipulateAsync(
    imageUri,
    [{ resize: { width: INPUT_SIZE, height: INPUT_SIZE } }],
    {
      compress: 1,
      format: SaveFormat.JPEG,
      base64: true,
    },
  );

  if (!resized.base64) {
    throw new Error("Image preprocessing failed.");
  }

  const jpegBytes = toByteArray(resized.base64);
  const decoded = jpeg.decode(jpegBytes, { useTArray: true, formatAsRGBA: true });
  if (decoded.width !== INPUT_SIZE || decoded.height !== INPUT_SIZE) {
    throw new Error("Unexpected preprocessed image size.");
  }

  const out = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  const rgba = decoded.data;
  const totalPixels = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < totalPixels; i += 1) {
    const src = i * 4;
    const dst = i * 3;
    out[dst] = rgba[src] / 255;
    out[dst + 1] = rgba[src + 1] / 255;
    out[dst + 2] = rgba[src + 2] / 255;
  }

  return out;
}

async function detectImage(
  model: TensorflowModel,
  imageUri: string,
): Promise<Detection[]> {
  const input = await imageToTensor(imageUri);
  const outputs = model.runSync([input]);
  if (!outputs.length) return [];

  const out = toFloatArray(outputs[0]);
  const parsed = postprocess(out);
  return parsed.map((d) => ({
    classId: d.cls,
    className: d.label,
    score: d.score,
    x1: d.x1,
    y1: d.y1,
    x2: d.x2,
    y2: d.y2,
  }));
}

export default function ResultScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    imageUri?: string;
  }>();
  const imageUri = Array.isArray(params.imageUri)
    ? params.imageUri[0]
    : params.imageUri;

  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [parsedDetections, setParsedDetections] = useState<Detection[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const loaded = await loadTensorflowModel(
          require("../../assets/models/best_float16 (2).tflite"),
        );
        if (!cancelled) setModel(loaded);
      } catch (error) {
        if (!cancelled) {
          console.warn("model_load_failed", error);
          setAnalysisError("Failed to load detection model.");
          setIsAnalyzing(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!imageUri) {
      setParsedDetections([]);
      setAnalysisError("No image provided.");
      setIsAnalyzing(false);
      return;
    }

    if (!model) {
      setIsAnalyzing(true);
      return;
    }

    let cancelled = false;
    setIsAnalyzing(true);
    setAnalysisError(null);

    (async () => {
      try {
        const next = await detectImage(model, imageUri);
        if (!cancelled) setParsedDetections(next);
      } catch (error) {
        if (!cancelled) {
          console.warn("image_detect_failed", error);
          setParsedDetections([]);
          setAnalysisError("Failed to analyze image.");
        }
      } finally {
        if (!cancelled) setIsAnalyzing(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [imageUri, model]);

  const hasHazard = useMemo(
    () => parsedDetections.some((d) => d.className.startsWith("no ")),
    [parsedDetections],
  );
  const isError = analysisError !== null;

  const statusColor = isAnalyzing
    ? "#334155"
    : isError || hasHazard
      ? "#7f1d1d"
      : "#0f9d58";
  const summary = isAnalyzing
    ? "Analyzing image..."
    : isError
      ? analysisError
      : formatSummary(parsedDetections);
  const headerTitle = isAnalyzing
    ? "Analyzing..."
    : isError
      ? "Analysis Failed"
      : hasHazard
        ? "Hazard Detected"
        : "Safeguard";
  const cardTitle = isAnalyzing
    ? "Analyzing Image"
    : isError
      ? "Analysis Failed"
      : hasHazard
        ? "Missing PPE Detected"
        : "Congratulations!";
  const cardBody = isAnalyzing
    ? "Please wait while the image is being analyzed."
    : isError
      ? "We could not analyze this image. Try again with a clearer photo."
      : hasHazard
        ? "Some required PPE items were not detected."
        : "You have proper PPE. Keep going and be safe!";

  const now = new Date();
  const dateLabel = now.toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
    year: "numeric",
  });
  const timeLabel = now.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: statusColor }]}>
      <View style={styles.screen}>
        <View style={[styles.header, { backgroundColor: statusColor }]}>
          <Pressable style={styles.backButton} onPress={() => router.back()}>
            <Svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <Path
                d="M15 18l-6-6 6-6"
                stroke="#000"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
          </Pressable>
          <Text style={styles.headerTitle}>{headerTitle}</Text>
          <View style={styles.headerSpacer} />
        </View>

        <View style={styles.previewWrap}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="cover" />
          ) : (
            <View style={styles.previewEmpty}>
              <Text style={styles.previewEmptyText}>No image provided</Text>
            </View>
          )}

          {!isAnalyzing && !isError && (
            <Svg style={StyleSheet.absoluteFill}>
              {parsedDetections.map((d, idx) => {
                // The image resizeMode is "cover" inside a 320px height container.
                // Assuming screen width for X.
                const { width: screenW } = Dimensions.get("window");
                const previewSize = Math.max(screenW, 320);
                const offsetX = (screenW - previewSize) / 2;
                const offsetY = (320 - previewSize) / 2;

                const x1 = (d.x1 / INPUT_SIZE) * previewSize + offsetX;
                const y1 = (d.y1 / INPUT_SIZE) * previewSize + offsetY;
                const x2 = (d.x2 / INPUT_SIZE) * previewSize + offsetX;
                const y2 = (d.y2 / INPUT_SIZE) * previewSize + offsetY;
                const w = x2 - x1;
                const h = y2 - y1;

                if (w < 2 || h < 2) return null;

                let color = "#22c55e"; // Green
                if (d.className === "person") {
                  color = "#06b6d4"; // Cyan
                } else if (d.className.toLowerCase().includes("no ")) {
                  color = "#ef4444"; // Red
                }

                return (
                  <G key={idx}>
                    <Rect
                      x={x1}
                      y={y1}
                      width={w}
                      height={h}
                      stroke={color}
                      strokeWidth={3}
                      fill="transparent"
                      rx={4}
                      ry={4}
                    />
                    <Rect
                      x={x1}
                      y={Math.max(0, y1 - 20)}
                      width={Math.min(w, 100)}
                      height={20}
                      fill={color}
                      rx={4}
                      ry={4}
                    />
                    <SvgText
                      x={x1 + 4}
                      y={Math.max(14, y1 - 6)}
                      fill="white"
                      fontSize={12}
                      fontWeight="bold"
                    >
                      {d.className}
                    </SvgText>
                  </G>
                );
              })}
            </Svg>
          )}
          <View
            style={[
              styles.frame,
              {
                borderColor: isAnalyzing
                  ? "rgba(148, 163, 184, 0.9)"
                  : isError || hasHazard
                    ? "rgba(239, 68, 68, 0.9)"
                    : "rgba(22, 163, 74, 0.9)",
              },
            ]}
          />
          <View
            style={[
              styles.summaryPill,
              {
                backgroundColor: isAnalyzing
                  ? "#334155"
                  : isError || hasHazard
                    ? "#b91c1c"
                    : "#16a34a",
              },
            ]}
          >
            <Text style={styles.summaryText}>{summary}</Text>
          </View>
        </View>

        <View style={styles.metaRow}>
          <Text style={styles.metaText}>{dateLabel}</Text>
          <View style={styles.metaDot} />
          <Text style={styles.metaText}>{timeLabel}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>{cardTitle}</Text>
          <Text style={styles.cardBody}>{cardBody}</Text>
        </View>

        {parsedDetections.length > 0 && (
          <View style={styles.logCard}>
            <Text style={styles.logHeader}>DETECTION LOG</Text>
            <ScrollView style={styles.logScroll} contentContainerStyle={styles.logContent}>
              {parsedDetections.map((d, idx) => (
                <View key={idx} style={styles.logItem}>
                  <View style={[styles.logIndicator, { backgroundColor: d.className === 'person' ? '#06b6d4' : (d.className.startsWith('no ') ? '#ef4444' : '#22c55e') }]} />
                  <Text style={styles.logText}>{toTitleCase(d.className)}</Text>
                  <Text style={styles.logScore}>{(d.score * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </ScrollView>
          </View>
        )}

        <Pressable
          style={[
            styles.primaryButton,
            { backgroundColor: isAnalyzing ? "#334155" : "#1f2937" },
          ]}
          onPress={() => router.back()}
        >
          <Text style={styles.primaryButtonText}>Done</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  screen: {
    flex: 1,
    backgroundColor: "#0f172a",
    paddingBottom: 24,
  },
  header: {
    paddingHorizontal: 18,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#ffffff",
    alignItems: "center",
    justifyContent: "center",
  },
  backIcon: {
    color: "#f8fafc",
    fontSize: 20,
  },
  headerTitle: {
    color: "#f8fafc",
    fontSize: 16,
    fontWeight: "700",
  },
  headerSpacer: {
    width: 36,
  },
  previewWrap: {
    borderRadius: 20,
    overflow: "hidden",
    marginTop: 10,
    backgroundColor: "#111827",
    height: 320,
  },
  preview: {
    width: "100%",
    height: "100%",
  },
  previewEmpty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  previewEmptyText: {
    color: "#94a3b8",
  },
  frame: {
    position: "absolute",
    left: 18,
    right: 18,
    top: 24,
    bottom: 24,
    borderWidth: 2,
    borderRadius: 18,
  },
  summaryPill: {
    position: "absolute",
    top: 16,
    left: 18,
    right: 18,
    borderRadius: 16,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  summaryText: {
    color: "#f8fafc",
    fontSize: 12,
    fontWeight: "700",
    textAlign: "center",
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 14,
  },
  metaText: {
    color: "#cbd5f5",
    fontSize: 12,
  },
  metaDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#64748b",
    marginHorizontal: 8,
  },
  card: {
    marginTop: 16,
    backgroundColor: "#111827",
    borderRadius: 16,
    padding: 16,
  },
  cardTitle: {
    color: "#f8fafc",
    fontSize: 16,
    fontWeight: "700",
  },
  cardBody: {
    color: "#94a3b8",
    marginTop: 6,
    fontSize: 13,
  },
  primaryButton: {
    marginTop: 18,
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: "center",
    marginHorizontal: 18,
  },
  primaryButtonText: {
    color: "#f8fafc",
    fontWeight: "700",
  },
  logCard: {
    marginHorizontal: 18,
    marginTop: 16,
    backgroundColor: "#111827",
    borderRadius: 16,
    padding: 16,
    flex: 1,
    maxHeight: 250,
  },
  logHeader: {
    color: "#64748b",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1,
    marginBottom: 10,
  },
  logScroll: {
    flex: 1,
  },
  logContent: {
    paddingBottom: 10,
  },
  logItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#1e293b",
  },
  logIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  logText: {
    color: "#f8fafc",
    fontSize: 14,
    fontWeight: "500",
    flex: 1,
  },
  logScore: {
    color: "#94a3b8",
    fontSize: 12,
    fontFamily: Platform.select({ ios: "Courier", android: "monospace" }),
  },
});
