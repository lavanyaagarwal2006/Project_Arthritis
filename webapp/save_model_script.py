"""
Helper script to add to your Jupyter notebook to save the trained model

Add this code cell at the end of your Image_analysis_multiclass_balanced.ipynb
notebook after training is complete.
"""

# Save the trained model
model.save('arthritis_model.h5')
print("=" * 60)
print("Model saved successfully as 'arthritis_model.h5'")
print("=" * 60)
print("\nNext steps:")
print("1. Move the model file to the webapp directory:")
print("   mv arthritis_model.h5 ../webapp/")
print("\n2. Navigate to the webapp directory:")
print("   cd ../webapp/")
print("\n3. Install dependencies:")
print("   pip install -r requirements.txt")
print("\n4. Run the web application:")
print("   python app.py")
print("\n5. Open your browser to:")
print("   http://localhost:5000")
print("=" * 60)
