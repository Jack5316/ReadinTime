import { FC, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Home: FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    navigate("/library");
  }, [navigate]);

  return null;
};

export default Home;
