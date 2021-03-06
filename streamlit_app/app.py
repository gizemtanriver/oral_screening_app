import streamlit as st
from PIL import Image
import numpy as np
from detect import yolov5

def main():
    
    classify_bb = True
    augment = False
    
    ###############
    # App layout
    ###############
    # Add a title 
    st.title("Oral Lesion Detection and Classification")
    
    # Add a sidebar
    st.sidebar.markdown("# Detection Model Parameters")
    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.05)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.05)
    
    # silence deprecation warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
        
    ###############
    # Predictions
    ###############
    file_up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file_up is not None:
        # Show uploaded image
        image = Image.open(file_up)
        # st.image(image, caption='Uploaded Image.', use_column_width=True) # Show original image
        # st.write("")
        st.write("Processing the input...")
        
        # Make predictions 
        try:
            output, output_clf = yolov5(file_up, confidence_threshold, overlap_threshold, classify_bb, augment)
        except:
            print("Encountered an issue while processing. Please try again.")
        
        # Draw the header and image.
        st.subheader("Real-time detections")
        st.markdown("**Detector** (overlap threshold `%1.2f`) (confidence threshold `%1.2f`)" % (overlap_threshold, confidence_threshold))
        st.write("- Overlap (non-max suppression) threshold: Used for suppressing false-positive predictions. Discards a prediction that overlaps with the top-scoring prediction higher than this threshold. Increase the value to visualize more detections (if exists).")
        st.write("- Confidence threshold: Filters out a prediction that has confidence score (probability of the object class (i.e. lesion) appearing in the bounding box) lower than this threshold. Lower the value to visualize more detections.")
        
        st.image(output.astype(np.uint8), use_column_width=True)
        
        if output_clf!=None:
            st.markdown("**Classifier Results**")
            # print out the prediction labels with scores
            for i, pred in enumerate(output_clf, 1):
                st.write("Detected Lesion #{}".format(i))
                for score in pred:
                    st.write(" - {}, Score: {} %".format(score[0], round(score[1],2)))



if __name__ == "__main__":
    main()