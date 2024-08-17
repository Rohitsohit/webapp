import React, { useState } from 'react';
import axios from 'axios';

function ImageUpload() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');
    const [caption, setcaption] = useState();

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = () => {
        if (!selectedFile) {
            setUploadStatus('Please select an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedFile);

        axios.post('http://127.0.0.1:5000/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        })
        .then(response => {
            setUploadStatus('Image uploaded successfully!');
            console.log('File Path:', response.data.caption);
            setcaption(response.data.caption);
        })
        .catch(error => {
            setUploadStatus('Failed to upload image.');
            console.error('Error:', error);
        });
    };

    return (
        <div>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload Image</button>
            <p>{uploadStatus}</p>
            <h1>{caption}</h1>
        </div>
    );
}

export default ImageUpload;
