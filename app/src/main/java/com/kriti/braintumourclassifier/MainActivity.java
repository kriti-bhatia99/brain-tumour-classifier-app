package com.kriti.braintumourclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.kriti.braintumourclassifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    ImageView imageView; TextView outputTextView;
    Button selectImageButton, classifyImageButton;
    Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        outputTextView = findViewById(R.id.outputTextView);
        selectImageButton = findViewById(R.id.selectImageButton);
        classifyImageButton = findViewById(R.id.classifyImageButton);

        selectImageButton.setOnClickListener(selectImageListener);
        classifyImageButton.setOnClickListener(classifyImageListener);
    }

    View.OnClickListener selectImageListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, 1);
        }
    };

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1)
        {
            imageView.setImageURI(data.getData());
            Uri uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    View.OnClickListener classifyImageListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            try {
                // Normalising the image
                bitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true);
                Model model = Model.newInstance(getApplicationContext());

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);
                TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                tensorImage.load(bitmap);
                ByteBuffer byteBuffer = tensorImage.getBuffer();
                inputFeature0.loadBuffer(byteBuffer);

                // Runs model inference and gets result.
                Model.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                // Releases model resources if no longer used.
                model.close();
                int maxIndex = 0;
                String outputText = "Prediction: ";

                for (int i=0; i<outputFeature0.getFloatArray().length; i++)
                {
                    if (outputFeature0.getFloatArray()[i] > outputFeature0.getFloatArray()[maxIndex])
                    {
                        maxIndex = i;
                    }
                }

                switch(maxIndex)
                {
                    case 0:
                        outputText += "No Tumour";
                        break;
                    case 1:
                        outputText += "Glioma Tumour";
                        break;
                    case 2:
                        outputText += "Meningioma Tumour";
                        break;
                    case 3:
                        outputText += "Pituitary Tumour";
                }

                outputTextView.setText(outputText);
            }
            catch (NullPointerException e)
            {
                Toast.makeText(MainActivity.this, "Please select an image", Toast.LENGTH_SHORT).show();
            }
            catch (IOException e)
            {
                Toast.makeText(MainActivity.this, "Error", Toast.LENGTH_SHORT).show();
                e.printStackTrace();
            }
        }
    };
}