import { redirect } from "next/navigation";

export default function DashboardIndex() {
  // Redirect to a default prediction dashboard (Wheat / Indore)
  redirect("/dashboard/wheat/Indore Mandi");
}
