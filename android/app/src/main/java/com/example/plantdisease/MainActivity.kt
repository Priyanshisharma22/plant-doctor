package com.example.plantdisease

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private var classifier: Classifier? = null
    private lateinit var preview: ImageView
    private lateinit var progress: ProgressBar
    private lateinit var tvResult: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvWarning: TextView
    private lateinit var tvTreatment: TextView
    private var cameraUri: Uri? = null

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { analyse(it) }
    }
    private val takePhoto = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) cameraUri?.let { analyse(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.imagePreview)
        progress = findViewById(R.id.progress)
        tvResult = findViewById(R.id.tvResult)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvWarning = findViewById(R.id.tvWarning)
        tvTreatment = findViewById(R.id.tvTreatment)

        // If the model cannot be loaded, show the reason on screen instead of crashing.
        try {
            classifier = Classifier(this)
        } catch (e: Throwable) {
            tvResult.text = "Model failed to load"
            tvWarning.text = "${e.javaClass.simpleName}: ${e.message}"
        }

        findViewById<Button>(R.id.btnGallery).setOnClickListener { pickImage.launch("image/*") }
        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            val file = File(cacheDir, "leaf_${System.currentTimeMillis()}.jpg")
            cameraUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            takePhoto.launch(cameraUri!!)
        }
    }

    private fun analyse(uri: Uri) {
        val clf = classifier
        if (clf == null) {
            Toast.makeText(this, "Model is not loaded", Toast.LENGTH_SHORT).show()
            return
        }
        progress.visibility = View.VISIBLE
        lifecycleScope.launch {
            try {
                val (bmp, pred) = withContext(Dispatchers.Default) {
                    val b = loadBitmap(uri)
                    b to clf.classify(b)
                }
                preview.setImageBitmap(bmp)
                tvResult.text = if (pred.healthy) "${pred.plant}: Healthy" else "${pred.plant}: ${pred.condition}"
                tvConfidence.text = "Confidence: %.1f%%".format(pred.confidence * 100)
                tvWarning.text = if (pred.confidence < 0.6f)
                    "Low confidence - retake the photo in good light, close to a single leaf." else ""
                tvTreatment.text = buildString {
                    append("Treatment\n${pred.treatment}")
                    if (pred.prevention.isNotEmpty()) append("\n\nPrevention\n${pred.prevention}")
                }
            } catch (e: Exception) {
                tvResult.text = "Could not analyse image"
                tvWarning.text = "${e.javaClass.simpleName}: ${e.message}"
            } finally {
                progress.visibility = View.GONE
            }
        }
    }

    /** Decode a downsampled bitmap and fix its EXIF rotation (camera photos are often rotated). */
    private fun loadBitmap(uri: Uri): Bitmap {
        val opts = BitmapFactory.Options().apply { inSampleSize = 2 }
        val bmp = contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, opts) }
            ?: error("decode failed")
        val orientation = contentResolver.openInputStream(uri).use {
            ExifInterface(it!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        val degrees = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (degrees == 0f) return bmp
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(degrees) }, true)
    }
}