from flask import Flask, render_template, request, url_for
import pd1
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import randomcolor
import creategif as cg


app = Flask(__name__, template_folder='./templates', static_folder='./templates')

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/test', methods=['POST', 'GET'])
def test():
    dim = int(request.form['input1'])
    s_shape = int(request.form['input2'])
    s_step = int(request.form['input3'])
    data_pts = pd1.mainfunction(dim, s_shape, s_step)
    list_of_planes = data_pts[0]
    data = data_pts[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('<--- X --->')
    ax.set_ylabel('<--- Y --->')
    ax.set_zlabel('<--- Z --->')
    ax.voxels(data, facecolors='grey', alpha=0.5, edgecolors=(1,1,1,0.5))

    for plane in list_of_planes:
        # color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        color = randomcolor.RandomColor().generate()
        ax.voxels(plane, facecolors=color[0], alpha=0.8, edgecolors='white')
    
    for ii in range(0, 360, 10):
        ax.view_init(azim = ii)
        fig.savefig(f'./moviefiles/movie{ii}.png')
    # for ii in range(0, 360,10):
    #     ax1.view_init(elev=10., azim=ii)
    #     plt.savefig(f"movie{ii}.png")
    
    cg.createanim()
    image_url = url_for('static', filename = 'animated_slow.gif')
    # plt.show()
    return render_template('index.html', image_url=image_url)

@app.route('/about')
def about_page():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
