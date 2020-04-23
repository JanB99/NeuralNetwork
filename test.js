let xs = [
    Matrix.fromArray([
        [0], [0]
    ]),
    Matrix.fromArray([
        [1], [0]
    ]),
    Matrix.fromArray([
        [0], [1]
    ]),
    Matrix.fromArray([
        [1], [1]
    ]),
]

let ys = [
    Matrix.fromArray([
        [0],
    ]),
    Matrix.fromArray([
        [1],
    ]),
    Matrix.fromArray([
        [1],
    ]),
    Matrix.fromArray([
        [0],
    ])
]

const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d');

const WIDTH = 800;
const HEIGHT = 800;
const scale = 10;

canvas.width = WIDTH
canvas.height = HEIGHT

let nn = new NeuralNetwork(2, [16], 1);

const animate = () => {

    nn.lr = document.getElementById('slider').value

    nn.train(xs, ys, 1000);
    for (let x = 0; x < WIDTH/scale; x++){
        for (let y = 0; y < WIDTH/scale; y++){
            let output = nn.predict(Matrix.fromArray([
                [x / (WIDTH/scale)], [y / (HEIGHT/scale)]
            ]))
            // ctx.fillStyle = `rgb(
            //     ${output.data[0][0] * 255},
            //     ${output.data[0][0] * 255},
            //     ${output.data[0][0] * 255})`;
            ctx.fillStyle = `hsl(
                ${output.data[0][0] * 360},
                ${output.data[0][0] * 100}%,
                ${output.data[0][0] * 100}%)`;
            ctx.fillRect(x * scale, y * scale, scale, scale)
        }   
    }

    requestAnimationFrame(animate)
}

animate();