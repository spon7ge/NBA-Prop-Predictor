import { Routes, Route } from "react-router-dom";
import App from "@/App";
import { HomePage } from "@/pages/HomePage";
import { LabPage } from "@/pages/LabPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/slate" element={<App />} />
      <Route path="/lab" element={<LabPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
