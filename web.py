from flask import Flask, escape, request, send_file, render_template
import display_image

app = Flask(__name__)

@app.route('/')
def hello():
    # images = display_image.load_images('data/full_numpy_bitmap_airplane.npy')
    # img = display_image.get_image_from_images(images, 0)
    # display_image.save_as_image(img)
    return render_template('index.html')
    # return send_file('image.png')

if __name__ == "__main__":
    app.run(debug=True)