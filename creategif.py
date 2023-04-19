from PIL import Image

# List of image file names
def createanim():
    image_list = []
    for i in range(0, 360, 10):
        image_list.append(f'./moviefiles/movie{i}.png')
        

    # print(image_list)

    # Open the first image
    im = Image.open(image_list[0])

    # Create a list to hold the frames for the GIF
    frames = []

    # Loop through each image in the list
    for image_name in image_list:
        # Open the image file
        im = Image.open(image_name)
        # Add the image to the list of frames
        frames.append(im.copy())

    # Save the frames as a GIF
    frames[0].save('./templates/animated_slow.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
