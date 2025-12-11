import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { GraduationCap, Briefcase } from "lucide-react"

const timelineData = [
  {
    type: "work",
    title: "|Intern",
    company: "Virtusa",
    period: "June-August 2025",
    description:
      "Developed and Implemented Predictive Maintenance of Banking Infrastructure for real-time monitoring and analytics platform that leverages machine learning to assess the health of banking Infrastructure.Enhanced the UI,Implemented SMTP for Customized Mail Notifications,Integrated JWT,Created Docker Containers",
    technologies: ["Python", "MongoDB", "React", "Docker","Machine Learning Algorithms"],
  },
  {
    type: "work",
    title: "Technical Trianee",
    company: "TEKsystems Global Services",
    period: "April-June 2025",
    description:
      "Developed a Full Stack Website JUSFurnishIT ,an innovative online platform that connects interior designers with customers seeking personalized home decor solutions, enabling CRUD operations using mvc architecture.Designed RESTful API endpoints with Spring Data JPA,Integrated JWT role based authentication,Dockerization.",
    technologies: ["Spring Boot", "SQL", "React", "Docker","RESTful APIs"],
  },
  {
    type: "education",
    title: "Bachelor of Computer Science",
    company: "CMR Institute of Technology",
    period: "2021 - 2025",
    description:
      "Foundation in computer science principles, programming languages, and software development lifecycle.",
    technologies: ["Java", "C++", "Database Design", "Web Development"],
  },
];

export function TimelineSection() {
  return (
    <section className="py-16">
      <h2 className="text-3xl font-bold text-center mb-12">
        Experience & Education
      </h2>
      <div className="relative">
        {/* Timeline Line */}
        <div className="absolute left-4 md:left-1/2 top-0 bottom-0 w-0.5 bg-primary/20 transform md:-translate-x-0.5" />

        <div className="space-y-8">
          {timelineData.map((item, index) => (
            <div
              key={index}
              className={`relative flex items-center ${
                index % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"
              }`}
            >
              {/* Timeline Dot */}
              <div className="absolute left-4 md:left-1/2 w-3 h-3 bg-primary rounded-full transform -translate-x-1.5 md:-translate-x-1.5 z-10">
                <div className="absolute inset-0 bg-primary rounded-full animate-ping opacity-20" />
              </div>

              {/* Content */}
              <div
                className={`ml-12 md:ml-0 md:w-1/2 ${
                  index % 2 === 0 ? "md:pr-8" : "md:pl-8"
                }`}
              >
                <Card className="bg-card/50 backdrop-blur-sm border-primary/10 transition-all hover:-translate-y-2 duration-700 hover:ease-in-out">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-2 mb-2">
                      {item.type === "work" ? (
                        <Briefcase className="w-5 h-5 text-primary" />
                      ) : (
                        <GraduationCap className="w-5 h-5 text-primary" />
                      )}
                      <Badge
                        variant={item.type === "work" ? "default" : "secondary"}
                      >
                        {item.type === "work" ? "Work" : "Education"}
                      </Badge>
                    </div>
                    <h3 className="text-xl font-semibold mb-1">{item.title}</h3>
                    <p className="text-primary font-medium mb-2">
                      {item.company}
                    </p>
                    <p className="text-base text-muted-foreground mb-3">
                      {item.period}
                    </p>
                    <p className="text-muted-foreground mb-4">
                      {item.description}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {item.technologies.map((tech) => (
                        <Badge key={tech} variant="outline" className="text-xs">
                          {tech}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
