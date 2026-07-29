import { Routes, Route } from "react-router-dom";
import App from "@/App";
import { LabPage } from "@/pages/LabPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/lab" element={<LabPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
