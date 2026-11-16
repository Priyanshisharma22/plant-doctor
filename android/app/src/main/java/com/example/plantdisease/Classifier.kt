package com.example.plantdisease

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class Prediction(
    val rawLabel: String,
    val plant: String,
    val condition: String,
    val confidence: Float,
    val healthy: Boolean,
    val treatment: String,
    val prevention: String,
)

/** Runs the on-device TFLite model. The model contains its own rescaling, so we pass raw 0-255 floats. */
class Classifier(context: Context) {
    private val size = 224
    private val interpreter: Interpreter
    private val labels: List<String>
    private val treatments: JSONObject

    init {
        val fd = context.assets.openFd("plant_disease.tflite")
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
        labels = context.assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        treatments = JSONObject(context.assets.open("treatments.json").bufferedReader().readText())
    }

    fun classify(bitmap: Bitmap): Prediction {
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val buf = ByteBuffer.allocateDirect(4 * size * size * 3).order(ByteOrder.nativeOrder())
        val px = IntArray(size * size)
        scaled.getPixels(px, 0, size, 0, 0, size, size)
        for (p in px) {
            buf.putFloat(((p shr 16) and 0xFF).toFloat()) // R
            buf.putFloat(((p shr 8) and 0xFF).toFloat())  // G
            buf.putFloat((p and 0xFF).toFloat())          // B
        }
        buf.rewind()

        val out = Array(1) { FloatArray(labels.size) }
        interpreter.run(buf, out)
        val probs = out[0]
        val best = probs.indices.maxByOrNull { probs[it] } ?: 0
        val raw = labels[best]
        val (plant, cond) = raw.split("___").let {
            clean(it[0]) to clean(it.getOrElse(1) { "" })
        }
        val info = treatments.optJSONObject(raw)
        return Prediction(
            rawLabel = raw, plant = plant, condition = cond, confidence = probs[best],
            healthy = raw.contains("healthy", ignoreCase = true),
            treatment = info?.optString("treatment") ?: "No treatment data available.",
            prevention = info?.optString("prevention") ?: "",
        )
    }

    private fun clean(s: String) = s.replace("_", " ").replace(",", "").trim()
}
