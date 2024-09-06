const { spawn } = require('child_process');
const path = require('path');

exports.handler = async function(event, context) {
  const process = spawn('python', [path.join(__dirname, 'streamlit_app.py')]);

  return new Promise((resolve, reject) => {
    let output = '';

    process.stdout.on('data', (data) => {
      output += data.toString();
    });

    process.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
    });

    process.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Process exited with code ${code}`));
      }
      resolve({
        statusCode: 200,
        body: output
      });
    });
  });
};
