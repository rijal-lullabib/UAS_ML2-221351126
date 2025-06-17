package com.example.studentpredictor

import android.annotation.SuppressLint
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.textfield.TextInputEditText
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Dropdowns
        val spinnerSchool = findViewById<AutoCompleteTextView>(R.id.spinnerSchool)
        val spinnerGender = findViewById<AutoCompleteTextView>(R.id.spinnerGender)

        val schoolAdapter = ArrayAdapter(
            this,
            android.R.layout.simple_dropdown_item_1line,
            resources.getStringArray(R.array.schools)
        )
        spinnerSchool.setAdapter(schoolAdapter)

        val genderAdapter = ArrayAdapter(
            this,
            android.R.layout.simple_dropdown_item_1line,
            resources.getStringArray(R.array.genders)
        )
        spinnerGender.setAdapter(genderAdapter)

        // Input fields
        val inputAge = findViewById<TextInputEditText>(R.id.inputAge)
        val inputStudy = findViewById<TextInputEditText>(R.id.inputStudy)
        val inputFailures = findViewById<TextInputEditText>(R.id.inputFailures)
        val inputG1 = findViewById<TextInputEditText>(R.id.inputG1)
        val inputG2 = findViewById<TextInputEditText>(R.id.inputG2)
        val inputG3 = findViewById<TextInputEditText>(R.id.inputG3)

        val textResult = findViewById<TextView>(R.id.textResult)
        val btnPredict = findViewById<Button>(R.id.btnPredict)

        // Load TFLite model
        interpreter = Interpreter(loadModelFile())

        btnPredict.setOnClickListener {
            try {
                val input = floatArrayOf(
                    resources.getStringArray(R.array.schools).indexOf(spinnerSchool.text.toString()).toFloat(),
                    resources.getStringArray(R.array.genders).indexOf(spinnerGender.text.toString()).toFloat(),
                    inputAge.text.toString().toFloatOrNull() ?: 0f,
                    inputStudy.text.toString().toFloatOrNull() ?: 0f,
                    inputFailures.text.toString().toFloatOrNull() ?: 0f,
                    inputG1.text.toString().toFloatOrNull() ?: 0f,
                    inputG2.text.toString().toFloatOrNull() ?: 0f,
                    inputG3.text.toString().toFloatOrNull() ?: 0f
                )

                val inputBuffer = ByteBuffer.allocateDirect(4 * input.size).order(ByteOrder.nativeOrder())
                input.forEach { inputBuffer.putFloat(it) }

                val outputBuffer = ByteBuffer.allocateDirect(4 * 3).order(ByteOrder.nativeOrder())
                interpreter.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()

                val resultArray = FloatArray(3)
                for (i in resultArray.indices) {
                    resultArray[i] = outputBuffer.float
                }

                val maxIndex = resultArray.indices.maxByOrNull { resultArray[it] } ?: -1

                val labelNames = listOf("Rendah", "Sedang", "Tinggi")
                val outputText = buildString {
                    append("Hasil prediksi:\n")
                    resultArray.forEachIndexed { index, value ->
                        append("${labelNames[index]}: %.2f\n".format(value))
                    }
                    append("Prediksi akhir: ${labelNames[maxIndex]}")
                }

                textResult.text = outputText

            } catch (e: Exception) {
                textResult.text = "Error saat prediksi: ${e.message}"
            }
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("student_model.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}
