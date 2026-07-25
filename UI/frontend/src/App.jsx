import { useState } from "react";
import "./App.css";

function App() {
  // ============================================================
  // FORM STATE
  // ============================================================

  const [formData, setFormData] = useState({
    longitude: "",
    latitude: "",
    housing_median_age: "",
    total_rooms: "",
    total_bedrooms: "",
    population: "",
    households: "",
    median_income: "",
    ocean_proximity: "<1H OCEAN",
  });

  // ============================================================
  // OTHER STATES
  // ============================================================

  const [prediction, setPrediction] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  // ============================================================
  // HANDLE INPUT CHANGES
  // ============================================================

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previousData) => ({
      ...previousData,
      [name]: value,
    }));
  };

  // ============================================================
  // RESET FORM
  // ============================================================

  const handleReset = () => {
    setFormData({
      longitude: "",
      latitude: "",
      housing_median_age: "",
      total_rooms: "",
      total_bedrooms: "",
      population: "",
      households: "",
      median_income: "",
      ocean_proximity: "<1H OCEAN",
    });

    setPrediction(null);

    setError("");
  };

  // ============================================================
  // SEND DATA TO FLASK
  // ============================================================

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);

    setPrediction(null);

    setError("");

    try {
      // Convert numerical values from strings to numbers
      const dataToSend = {
        longitude: Number(formData.longitude),

        latitude: Number(formData.latitude),

        housing_median_age: Number(formData.housing_median_age),

        total_rooms: Number(formData.total_rooms),

        total_bedrooms: Number(formData.total_bedrooms),

        population: Number(formData.population),

        households: Number(formData.households),

        median_income: Number(formData.median_income),

        ocean_proximity: formData.ocean_proximity,
      };

      // Send request to Flask backend
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify(dataToSend),
      });

      // Convert Flask response to JSON
      const result = await response.json();

      // Check for backend errors
      if (!response.ok || !result.success) {
        throw new Error(result.error || "Unable to predict house price.");
      }

      // Save prediction
      setPrediction(result.predicted_price);
    } catch (error) {
      console.error("Prediction error:", error);

      setError(error.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // FORMAT PRICE
  // ============================================================

  const formatPrice = (price) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(price);
  };

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app">
      {/* ======================================================
          NAVBAR
      ======================================================= */}

      <nav className="navbar">
        <div className="navbar-content">
          <div className="logo">HomePredict</div>

          <div className="model-badge">Machine Learning</div>
        </div>
      </nav>

      {/* ======================================================
          MAIN CONTENT
      ======================================================= */}

      <main className="main-content">
        <div className="hero">
          <p className="eyebrow">AI-POWERED REAL ESTATE</p>

          <h1>House Price Predictor</h1>

          <p className="hero-description">
            Enter the characteristics of a residential property and our machine
            learning model will estimate its market value.
          </p>
        </div>

        {/* ==================================================
            PREDICTION FORM
        =================================================== */}

        <div className="prediction-card">
          <div className="card-header">
            <h2>Property Information</h2>

            <p>Enter the details of the property below.</p>
          </div>

          <form onSubmit={handleSubmit}>
            {/* ==============================================
                LOCATION SECTION
            =============================================== */}

            <div className="section">
              <h3>Location</h3>

              <div className="form-grid">
                {/* Longitude */}

                <div className="form-group">
                  <label htmlFor="longitude">Longitude</label>

                  <input
                    id="longitude"
                    type="number"
                    step="any"
                    name="longitude"
                    value={formData.longitude}
                    onChange={handleChange}
                    placeholder="-122.23"
                    required
                  />

                  <span className="input-help">Example: -122.23</span>
                </div>

                {/* Latitude */}

                <div className="form-group">
                  <label htmlFor="latitude">Latitude</label>

                  <input
                    id="latitude"
                    type="number"
                    step="any"
                    name="latitude"
                    value={formData.latitude}
                    onChange={handleChange}
                    placeholder="37.88"
                    required
                  />

                  <span className="input-help">Example: 37.88</span>
                </div>

                {/* Ocean Proximity */}

                <div className="form-group full-width">
                  <label htmlFor="ocean_proximity">Ocean Proximity</label>

                  <select
                    id="ocean_proximity"
                    name="ocean_proximity"
                    value={formData.ocean_proximity}
                    onChange={handleChange}
                    required
                  >
                    <option value="<1H OCEAN">
                      Less than 1 hour from ocean
                    </option>

                    <option value="INLAND">Inland</option>

                    <option value="NEAR OCEAN">Near Ocean</option>

                    <option value="NEAR BAY">Near Bay</option>

                    <option value="ISLAND">Island</option>
                  </select>
                </div>
              </div>
            </div>

            {/* ==============================================
                PROPERTY SECTION
            =============================================== */}

            <div className="section">
              <h3>Property Details</h3>

              <div className="form-grid">
                {/* Housing Age */}

                <div className="form-group">
                  <label htmlFor="housing_median_age">Housing Median Age</label>

                  <input
                    id="housing_median_age"
                    type="number"
                    min="0"
                    name="housing_median_age"
                    value={formData.housing_median_age}
                    onChange={handleChange}
                    placeholder="25"
                    required
                  />
                </div>

                {/* Total Rooms */}

                <div className="form-group">
                  <label htmlFor="total_rooms">Total Rooms</label>

                  <input
                    id="total_rooms"
                    type="number"
                    min="0"
                    name="total_rooms"
                    value={formData.total_rooms}
                    onChange={handleChange}
                    placeholder="5000"
                    required
                  />
                </div>

                {/* Total Bedrooms */}

                <div className="form-group">
                  <label htmlFor="total_bedrooms">Total Bedrooms</label>

                  <input
                    id="total_bedrooms"
                    type="number"
                    min="0"
                    name="total_bedrooms"
                    value={formData.total_bedrooms}
                    onChange={handleChange}
                    placeholder="1000"
                    required
                  />
                </div>

                {/* Population */}

                <div className="form-group">
                  <label htmlFor="population">Population</label>

                  <input
                    id="population"
                    type="number"
                    min="0"
                    name="population"
                    value={formData.population}
                    onChange={handleChange}
                    placeholder="1500"
                    required
                  />
                </div>

                {/* Households */}

                <div className="form-group">
                  <label htmlFor="households">Households</label>

                  <input
                    id="households"
                    type="number"
                    min="0"
                    name="households"
                    value={formData.households}
                    onChange={handleChange}
                    placeholder="500"
                    required
                  />
                </div>

                {/* Median Income */}

                <div className="form-group">
                  <label htmlFor="median_income">Median Income</label>

                  <input
                    id="median_income"
                    type="number"
                    min="0"
                    step="any"
                    name="median_income"
                    value={formData.median_income}
                    onChange={handleChange}
                    placeholder="5.0"
                    required
                  />

                  <span className="input-help">
                    Median income in tens of thousands
                  </span>
                </div>
              </div>
            </div>

            {/* ==============================================
                BUTTONS
            =============================================== */}

            <div className="button-container">
              <button
                type="button"
                className="reset-button"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>

              <button
                type="submit"
                className="predict-button"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Predicting...
                  </>
                ) : (
                  "Predict House Price"
                )}
              </button>
            </div>
          </form>

          {/* =================================================
              ERROR MESSAGE
          ================================================== */}

          {error && (
            <div className="error-message">
              <strong>Prediction Failed</strong>

              <p>{error}</p>
            </div>
          )}

          {/* =================================================
              PREDICTION RESULT
          ================================================== */}

          {prediction !== null && (
            <div className="prediction-result">
              <div className="result-icon">$</div>

              <div className="result-content">
                <p className="result-label">Estimated House Value</p>

                <h2>{formatPrice(prediction)}</h2>

                <p className="result-note">
                  This estimate was generated by the trained machine learning
                  model.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* ======================================================
          FOOTER
      ======================================================= */}

      <footer>
        <p>House Price Prediction System</p>

        <p>Powered by Machine Learning & React</p>
      </footer>
    </div>
  );
}

export default App;
