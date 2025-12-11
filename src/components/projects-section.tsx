import { CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { GlareCard } from "@/components/ui/glare-card";
import {
  ImageIcon,
  Tablets,
  LibraryBig,
  MonitorPlayIcon as TvMinimalPlay,
  MapPinned,
  Star,
} from "lucide-react";

type ProjectStatus = "Completed" | "Ongoing";

interface Project {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  status: ProjectStatus;
  technologies: string[];
  link: string;
}

const projects: Project[] = [
  {
    icon: ImageIcon,
    title: "Lung Cancer Prediction Model",
    description:
      "A predictive model for early lung cancer detection using ReLU Activation Function.",
    status: "Completed",
    technologies: ["Python", "ReactJS", "AI/ML"],
    link: "https://gith",
  },
  {
    icon: Tablets,
    title: "JUSFurnishIT",
    description:
      "Full Stack Application that connects interior designers with customers.",
    status: "Completed",
    technologies: ["ReactJS", "SpringBoot", "SQL","API"],
    link: "https://github.com/karthik-51/JusFurnishIt",
  },
  {
    icon: LibraryBig,
    title: "Banking predictive analysis",
    description:
      "Real-time monitoring platform with ML to assess the health of banking Infrastructure.",
    status: "Ongoing",
    technologies: ["Python", "ML", "MongoDB", "Docker"],
    link: "https://github.com/karthik-51/VIRTUSA-JatayuS4-CodeCrusaders.git",
  },
  
];

export function ProjectsSection() {
  const statusColors: Record<ProjectStatus, string> = {
    Completed: "bg-green-500/90 dark:bg-green-700/90 text-white",
    Ongoing: "bg-orange-500/90 dark:bg-orange-400/90 text-white",
  };

  return (
    <section className="py-16 px-4">
      <div className="max-w-[864px] mx-auto">
        <h2 className="text-3xl font-bold text-center mb-12 text-slate-900 dark:text-slate-100">
          My Projects
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((activity, index) => {
            const Icon = activity.icon;
            return (
              <Link
                key={index}
                target="_blank"
                href={activity.link}
                className="block transition-transform hover:scale-[1.02] focus:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-lg"
              >
                <GlareCard className="flex flex-col h-full p-5 w-full bg-card dark:bg-slate-900">
                  <CardHeader className="p-0 mb-4">
                    <div className="flex items-start justify-between w-full">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Icon className="w-6 h-6 text-primary" />
                      </div>
                      <div className="text-right">
                        <CardTitle className="text-xl line-clamp-2 text-slate-900 dark:text-slate-100">
                          {activity.title}
                        </CardTitle>
                        <Badge
                          className={`mt-1 px-2 py-1 text-xs font-medium rounded-md border-0 ${
                            statusColors[activity.status]
                          }`}
                        >
                          {activity.status}
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0 mb-4 flex-grow">
                    <p className="text-slate-600 dark:text-slate-300 line-clamp-3">
                      {activity.description}
                    </p>
                  </CardContent>
                  <div className="flex flex-wrap gap-2 mt-auto">
                    {activity.technologies.map((tech) => (
                      <Badge
                        key={tech}
                        variant="secondary"
                        className="text-xs bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
                      >
                        {tech}
                      </Badge>
                    ))}
                  </div>
                </GlareCard>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
}
